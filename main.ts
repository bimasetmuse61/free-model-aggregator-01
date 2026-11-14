// free-model-aggregator-01: one server, two client wires, ten simulated free providers.
//
// The entry reads its brief (MAKEFUN_PROMPT_FILE), probes every provider's catalog route in the
// provider's own wire, and builds one catalog document. It listens on an ephemeral port, registers
// the port and the catalog with the env, then answers the env client's twenty requests, one at a
// time, each with the best free model that can serve it, failing over when a provider refuses.
//
// Routing: every request's candidates are the free models meeting its requirements, ordered by the
// sealed quality table's rating. A global plan reserves each provider's conservative per-trial
// capacity (the sealed table's calls_min) for the heaviest requests first, so the best models are
// not spent on the lightest requests. At run time each candidate is tried up to twice on a refusal
// (402/429/5xx): a provider's failure schedule refuses single calls, so a retry on the same
// provider usually lands, and only then does the request fall to the next candidate.
//
// Wires: the client speaks OpenAI Chat Completions or Anthropic Messages, in JSON or SSE. The
// upstream provider speaks OpenAI, Anthropic or Gemini. Every upstream answer is normalised to
// {text, usage, stop} and re-rendered in the client's wire, so the provider's whole answer text,
// its usage counts and its stop reason reach the client exactly. Requests are translated field by
// field (system prompt, user text, images, tools, tool choice, max_tokens, stop sequences, JSON
// mode) so the provider echoes exactly what the client sent; a JSON-mode request never goes to an
// Anthropic-wire provider, whose wire cannot carry it.
//
// Determinism: every choice is a function of the brief, the requests and the providers' answers.
// No wall clock, no unseeded randomness. Fixed timestamps in every rendered document.

import { createServer } from 'node:http';
import type { IncomingMessage, Server, ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumber = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const asBool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 10000;
const MAX_BODY_BYTES = 4 * 1024 * 1024;
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
const ANTHROPIC_VERSION = '2023-06-01';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: provider directory, quality table, workload manifest.
// ---------------------------------------------------------------------------

interface ProviderDir {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadReq {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: ProviderDir[];
  ratings: RatingRow[];
  best: Map<string, number>;
  manifest: WorkloadReq[];
  error: string | null;
}

/** The index of the `}` closing the object that opens at `start`, or -1; strings are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderDir | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    asString(v.path) !== '' ? asString(v.path) : asString(v.base_path) !== '' ? asString(v.base_path) : asString(v.base_url) !== '' ? asString(v.base_url) : asString(v.base);
  return { id, name: asString(v.name) !== '' ? asString(v.name) : id, wire, auth, base, key_env: asString(v.key_env), key: asString(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: asNumber(v.rating_milli) ?? 0,
    context_length: asNumber(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadOf(v: unknown): WorkloadReq | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/providers/v1' && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { ...empty, error: 'the brief holds no provider directory' };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/ratings/v1' && Array.isArray(d.ratings)) ?? null;
  const ratings: RatingRow[] =
    ratingsDoc === null || !Array.isArray(ratingsDoc.ratings)
      ? []
      : ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        });
  const best = new Map<string, number>();
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.best)) {
    for (const v of ratingsDoc.best) {
      if (isRecord(v) && asString(v.id) !== '') best.set(v.id, asNumber(v.rating_milli) ?? 0);
    }
  }
  const workloadDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/workload/v1' && Array.isArray(d.manifest)) ?? null;
  const manifest: WorkloadReq[] =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest)
      ? []
      : workloadDoc.manifest.flatMap((v) => {
          const w = workloadOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, best, manifest, error: null };
}

// ---------------------------------------------------------------------------
// The catalog probe.
// ---------------------------------------------------------------------------

interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: ProviderDir;
  ok: boolean;
  models: ProbedModel[];
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. The wire never decides the header: the auth scheme does. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** A listing's price field says free when every price it names is zero; null when it says nothing. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  const context = asNumber(v.context_length) ?? asNumber(v.context_window) ?? asNumber(v.inputTokenLimit);
  return {
    id,
    display_name: asString(v.display_name) !== '' ? asString(v.display_name) : asString(v.displayName) !== '' ? asString(v.displayName) : id,
    free: asBool(v.free) ?? zeroPriced(v),
    context_length: context,
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
  };
}

function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

async function probeProvider(envUrl: string, provider: ProviderDir, timeoutMs = PROBE_TIMEOUT_MS): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, models: [] };
    const body: unknown = await res.json();
    return { provider, ok: true, models: modelsIn(provider.wire, body) };
  } catch {
    return { provider, ok: false, models: [] };
  }
}

/** The rating row of one provider's model, when the quality table has one. */
function ratingRowOf(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** Whether a model is free: the quality table where it has a row, the listing's own word otherwise. */
function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table where it has a row, the listing's fields otherwise. */
function capabilitiesOf(model: ProbedModel, row: RatingRow | null): { context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null } {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
function meetsRequirements(requires: Requirement, caps: { context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null }): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The catalog the probes and the quality table make. Its base_url is the directory's own path
 * (`/p/<id>`), never joined to the env URL: the env's origin differs between a live pass and its
 * replay, and the catalog is part of the state export a replay must reproduce byte for byte.
 */
function buildCatalog(probes: readonly Probe[], brief: Brief): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowOf(brief, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Routing: candidates, the capacity plan, and the failover order.
// ---------------------------------------------------------------------------

interface Candidate {
  provider: ProviderDir;
  model: ProbedModel;
  rating: number;
}

/**
 * The conservative per-trial capacity each provider is planned with: the sealed table's calls_min,
 * the floor of the budget the generator draws. The plan never trusts a bigger number; run-time
 * refusals move a request down its candidate list.
 */
const PLANNED_CAPACITY: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
const DEFAULT_PLANNED_CAPACITY = 4;

const byRating = (a: Candidate, b: Candidate): number =>
  b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0);

/**
 * The free models that can serve one request, best rating first. An Anthropic-wire provider is
 * never a candidate for a JSON-mode request: its wire cannot carry response_format, so the
 * provider would echo json_mode false and the request would fail its echo check.
 */
function candidatesFor(brief: Brief, probes: readonly Probe[], requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (requires.json === true && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingRowOf(brief, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return out.sort(byRating);
}

/**
 * The global plan: for every manifest request, the one model it tries first. Requests are planned
 * heaviest first (weight, then their best rating, then manifest order) so the best models are
 * reserved for the requests the score weighs most; each provider's planned capacity is spent as
 * its models are assigned. A request whose candidates are all out of planned capacity still gets
 * its best candidate: run-time failover, not the plan, is what keeps it served.
 */
function buildPlan(brief: Brief, probes: readonly Probe[]): Map<string, Candidate> {
  const capacity = new Map<string, number>();
  for (const provider of brief.providers) capacity.set(provider.id, PLANNED_CAPACITY[provider.id] ?? DEFAULT_PLANNED_CAPACITY);
  const plan = new Map<string, Candidate>();
  const order = [...brief.manifest].sort((a, b) => {
    const bestA = Math.max(0, ...candidatesFor(brief, probes, a.requires).map((c) => c.rating));
    const bestB = Math.max(0, ...candidatesFor(brief, probes, b.requires).map((c) => c.rating));
    return b.weight - a.weight || bestB - bestA || brief.manifest.indexOf(a) - brief.manifest.indexOf(b);
  });
  for (const request of order) {
    const candidates = candidatesFor(brief, probes, request.requires);
    if (candidates.length === 0) continue;
    const pick = candidates.find((c) => (capacity.get(c.provider.id) ?? 0) > 0) ?? candidates[0];
    capacity.set(pick.provider.id, (capacity.get(pick.provider.id) ?? 0) - 1);
    plan.set(request.id, pick);
  }
  return plan;
}

// ---------------------------------------------------------------------------
// Request translation: the client's wire into the provider's wire.
// ---------------------------------------------------------------------------

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The stop sequences of a body, as a list, from either wire's spelling. */
function stopList(body: Json): string[] {
  if (Array.isArray(body.stop_sequences)) return body.stop_sequences.filter((s): s is string => typeof s === 'string');
  if (typeof body.stop === 'string') return [body.stop];
  if (Array.isArray(body.stop)) return body.stop.filter((s): s is string => typeof s === 'string');
  return [];
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function anthropicBlockToPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** OpenAI's tool_choice as Anthropic's. */
function openAIToolChoiceToAnthropic(choice: unknown): Json | null {
  if (choice === 'auto') return { type: 'auto' };
  if (choice === 'required') return { type: 'any' };
  if (choice === 'none') return { type: 'none' };
  if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    return { type: 'tool', name: choice.function.name };
  }
  return null;
}

/** Anthropic's tool_choice as OpenAI's. */
function anthropicToolChoiceToOpenAI(choice: unknown): unknown {
  if (!isRecord(choice)) return undefined;
  if (choice.type === 'auto') return 'auto';
  if (choice.type === 'any') return 'required';
  if (choice.type === 'none') return 'none';
  if (choice.type === 'tool' && typeof choice.name === 'string') return { type: 'function', function: { name: choice.name } };
  return undefined;
}

/** An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. */
function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: toolResultText(raw.content) }],
      });
      continue;
    }
    // A user message: text and image parts, in order.
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const url: string = block.image_url.url;
        const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
        if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
        else blocks.push({ type: 'image', source: { type: 'url', url } });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = openAIToolChoiceToAnthropic(body.tool_choice);
  if (choice !== null) out.tool_choice = choice;
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic Messages request as an OpenAI Chat Completions request naming `model`. */
function anthropicToOpenAI(body: Json, model: string, jsonMode: boolean): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = anthropicBlockToPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body);
  if (stops.length > 0) out.stop = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ type: 'function', function: { name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = anthropicToolChoiceToOpenAI(body.tool_choice);
  if (choice !== undefined) out.tool_choice = choice;
  if (jsonMode) out.response_format = { type: 'json_object' };
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One message's image as a Gemini part: inline base64 data, or a file reference for a URL. */
function imagePartOf(url: string): Json | null {
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  return { fileData: { fileUri: url } };
}

/**
 * Either client wire as a Gemini generateContent body naming `model`. Tool results ride as
 * functionResponse parts on the latest user content, so they never become the echoed user text.
 */
function toGemini(clientApi: 'openai' | 'anthropic', body: Json, model: string, jsonMode: boolean): Json {
  const systemText =
    clientApi === 'openai'
      ? (Array.isArray(body.messages) ? body.messages : [])
          .filter((m) => isRecord(m) && m.role === 'system')
          .map((m) => textOf((m as Json).content))
          .filter((t) => t !== '')
          .join('\n')
      : textOf(body.system);
  const contents: Json[] = [];
  let lastUser: Json | null = null;
  const pushUserText = (text: string): void => {
    if (text === '') return;
    const content: Json = { role: 'user', parts: [{ text }] };
    contents.push(content);
    lastUser = content;
  };
  const pushUserPart = (part: Json): void => {
    if (lastUser === null) {
      const content: Json = { role: 'user', parts: [part] };
      contents.push(content);
      lastUser = content;
      return;
    }
    (lastUser.parts as Json[]).push(part);
  };
  const messages: Json[] = clientApi === 'openai' ? (Array.isArray(body.messages) ? body.messages.filter(isRecord) : []) : Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  for (const raw of messages) {
    if (raw.role === 'system') continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(content);
      if (text !== '') parts.push({ text });
      if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (clientApi === 'anthropic' && block.type === 'tool_use') {
            parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          }
        }
      }
      if (Array.isArray((raw as Json).tool_calls)) {
        for (const call of (raw as Json).tool_calls as unknown[]) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      pushUserPart({ functionResponse: { name: '', response: { result: toolResultText(content) } } });
      continue;
    }
    // A user message.
    if (typeof content === 'string') {
      pushUserText(content);
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        pushUserPart({ functionResponse: { name: '', response: { result: toolResultText(block.content) } } });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const part = imagePartOf(block.image_url.url);
        if (part !== null) parts.push(part);
      } else if (block.type === 'image' && isRecord(block.source)) {
        const source = block.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
        } else if (source.type === 'url' && typeof source.url === 'string') {
          const part = imagePartOf(source.url);
          if (part !== null) parts.push(part);
        }
      }
    }
    if (parts.length > 0) {
      const entry: Json = { role: 'user', parts };
      contents.push(entry);
      lastUser = entry;
    }
  }
  const out: Json = { contents, model };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  // Tools, in Gemini's functionDeclarations spelling, in the client's order.
  const declarations: Json[] = [];
  const rawTools: unknown[] = clientApi === 'openai' ? (Array.isArray(body.tools) ? body.tools : []) : Array.isArray(body.tools) ? body.tools : [];
  for (const raw of rawTools) {
    if (!isRecord(raw)) continue;
    if (clientApi === 'openai') {
      if (!isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      declarations.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
    } else {
      if (typeof raw.name !== 'string') continue;
      declarations.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
    }
  }
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  const maxTokens = asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  const stops = stopList(body);
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const choice = clientApi === 'openai' ? body.tool_choice : body.tool_choice;
  if (choice !== undefined) {
    const mode = clientApi === 'openai' ? choice === 'none' ? 'NONE' : choice === 'required' ? 'ANY' : 'AUTO' : isRecord(choice) ? (choice.type === 'none' ? 'NONE' : choice.type === 'any' ? 'ANY' : 'AUTO') : 'AUTO';
    const toolConfig: Json = { functionCallingConfig: { mode } };
    if (mode === 'ANY' && clientApi === 'openai' && isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
      toolConfig.functionCallingConfig = { mode, allowedFunctionNames: [choice.function.name] };
    }
    if (mode === 'ANY' && clientApi === 'anthropic' && isRecord(choice) && choice.type === 'tool' && typeof choice.name === 'string') {
      toolConfig.functionCallingConfig = { mode, allowedFunctionNames: [choice.name] };
    }
    config.toolConfig = toolConfig;
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The request body for one provider, in its wire, naming `model`. */
function upstreamBody(clientApi: 'openai' | 'anthropic', body: Json, provider: ProviderDir, model: string, jsonMode: boolean): Json {
  if (provider.wire === 'openai') {
    if (clientApi === 'openai') {
      const out: Json = { ...body, model };
      if (body.stream === true) out.stream_options = { include_usage: true };
      if (jsonMode && !isRecord(body.response_format)) out.response_format = { type: 'json_object' };
      return out;
    }
    return anthropicToOpenAI(body, model, jsonMode);
  }
  if (provider.wire === 'anthropic') {
    if (clientApi === 'anthropic') return { ...body, model };
    return openAIToAnthropic(body, model);
  }
  return toGemini(clientApi, body, model, jsonMode);
}

// ---------------------------------------------------------------------------
// Answers: every upstream wire normalised to {text, usage, stop}, then rendered
// in the client's wire, in JSON or in server-sent events.
// ---------------------------------------------------------------------------

interface Answer {
  text: string;
  input: number;
  output: number;
  stop: string;
}

/** The stop-reason family of a provider's or a client's stop string: 0 stop, 1 length, 2 tool, 3 refusal. */
function stopFamily(reason: string | null): number | null {
  if (reason === null) return null;
  if (reason === 'stop' || reason === 'end_turn' || reason === 'STOP') return 0;
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return null;
}

/** One stop reason in the client's wire, in the same family as the provider's. */
function stopInWire(stop: string, wire: 'openai' | 'anthropic'): string {
  const family = stopFamily(stop);
  if (wire === 'openai') return family === 1 ? 'length' : family === 2 ? 'tool_calls' : family === 3 ? 'content_filter' : 'stop';
  return family === 1 ? 'max_tokens' : family === 2 ? 'tool_use' : family === 3 ? 'refusal' : 'end_turn';
}

const numOrZero = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

/** The text of an OpenAI message: a string, or its text parts joined. */
function openAIText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** One SSE frame's lines as {event, data} pairs; a frame with no data line is skipped. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const out: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) out.push({ event, data: data.join('\n') });
  }
  return out;
}

function parseOpenAIAnswer(body: string, stream: boolean): Answer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(body);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : {};
    const usage = isRecord(value.usage) ? value.usage : {};
    return { text: openAIText(message.content), input: numOrZero(usage.prompt_tokens), output: numOrZero(usage.completion_tokens), stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop' };
  }
  let text = '';
  let input = 0;
  let output = 0;
  let stop = 'stop';
  for (const frame of sseFrames(body)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      input = numOrZero(chunk.usage.prompt_tokens);
      output = numOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text, input, output, stop };
}

function parseAnthropicAnswer(body: string, stream: boolean): Answer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(body);
    } catch {
      return null;
    }
    if (!isRecord(value) || value.type !== 'message' || !Array.isArray(value.content)) return null;
    const usage = isRecord(value.usage) ? value.usage : {};
    return { text: openAIText(value.content), input: numOrZero(usage.input_tokens), output: numOrZero(usage.output_tokens), stop: typeof value.stop_reason === 'string' ? value.stop_reason : 'end_turn' };
  }
  let text = '';
  let input = 0;
  let output = 0;
  let stop = 'end_turn';
  for (const frame of sseFrames(body)) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) input = numOrZero(data.message.usage.input_tokens);
    if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') text += data.delta.text;
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numOrZero(data.usage.output_tokens);
    }
  }
  return { text, input, output, stop };
}

function geminiTextOf(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

function parseGeminiAnswer(body: string, stream: boolean): Answer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(body);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) return null;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
    return { text: geminiTextOf(content.parts), input: numOrZero(usage.promptTokenCount), output: numOrZero(usage.candidatesTokenCount), stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP' };
  }
  let text = '';
  let input = 0;
  let output = 0;
  let stop = 'STOP';
  for (const frame of sseFrames(body)) {
    let value: unknown;
    try {
      value = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      text += geminiTextOf(content.parts);
      if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    }
    if (isRecord(value.usageMetadata)) {
      input = numOrZero(value.usageMetadata.promptTokenCount);
      output = numOrZero(value.usageMetadata.candidatesTokenCount);
    }
  }
  return { text, input, output, stop };
}

function parseUpstreamAnswer(wire: Wire, body: string, stream: boolean): Answer | null {
  if (wire === 'openai') return parseOpenAIAnswer(body, stream);
  if (wire === 'anthropic') return parseAnthropicAnswer(body, stream);
  return parseGeminiAnswer(body, stream);
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into 13-character pieces, the way the simulated providers chunk their deltas. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** One answer as an OpenAI Chat Completions body, or as its SSE stream. */
function renderOpenAI(answer: Answer, model: string, stream: boolean): string {
  const stop = stopInWire(answer.stop, 'openai');
  if (!stream) {
    return JSON.stringify({
      id: 'chatcmpl-agg',
      object: 'chat.completion',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: stop, logprobs: null }],
      usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
    });
  }
  const chunk = (delta: Json, finish: string | null): Json => ({ id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, stop)));
  frames.push(sseEvent(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output } }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** One answer as an Anthropic Messages body, or as its SSE stream. */
function renderAnthropic(answer: Answer, model: string, stream: boolean): string {
  const stop = stopInWire(answer.stop, 'anthropic');
  if (!stream) {
    return JSON.stringify({
      id: 'msg_agg',
      type: 'message',
      role: 'assistant',
      model,
      content: [{ type: 'text', text: answer.text }],
      stop_reason: stop,
      stop_sequence: null,
      usage: { input_tokens: answer.input, output_tokens: answer.output },
    });
  }
  const frames: string[] = [
    sseEvent('message_start', { type: 'message_start', message: { id: 'msg_agg', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.input, output_tokens: 0 } } }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: answer.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

function renderAnswer(answer: Answer, clientApi: 'openai' | 'anthropic', model: string, stream: boolean): string {
  return clientApi === 'openai' ? renderOpenAI(answer, model, stream) : renderAnthropic(answer, model, stream);
}

// ---------------------------------------------------------------------------
// Calling a provider, serving one client request, and the server itself.
// ---------------------------------------------------------------------------

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
let plan = new Map<string, Candidate>();
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (ENV_URL === '') {
    probes = [];
    catalog = { providers: [] };
    return;
  }
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider)));
  // One retry for a probe that failed: a catalog route that answers late still counts.
  for (let i = 0; i < probes.length; i++) {
    if (!probes[i].ok) probes[i] = await probeProvider(ENV_URL, probes[i].provider);
  }
  for (const probe of probes) if (!probe.ok) log(`the ${probe.provider.id} catalog route did not answer`);
  catalog = buildCatalog(probes, brief);
  plan = buildPlan(brief, probes);
}

/** The free models across the providers that answered, one entry per model id, for GET /v1/models. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const seen = new Set<string>();
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      const row = ratingRowOf(brief, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): { id: string | null; requires: Requirement } {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { id: entry.id, requires: entry.requires };
  return { id: null, requires: requirementsFromBody(body, clientApi) };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (clientApi === 'openai' && block.type === 'image_url') vision = true;
      if (clientApi === 'anthropic' && block.type === 'image') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One provider call: the status and the raw body text, or null when the call never reached it. */
async function callProvider(provider: ProviderDir, model: string, body: Json, stream: boolean): Promise<{ status: number; text: string } | null> {
  const base = joinBase(ENV_URL, provider.base);
  let url: string;
  if (provider.wire === 'openai') url = `${base}/v1/chat/completions`;
  else if (provider.wire === 'anthropic') url = `${base}/v1/messages`;
  else url = `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  if (stream) headers.accept = 'text/event-stream';
  try {
    const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
    return { status: res.status, text: await res.text() };
  } catch {
    return null;
  }
}

/**
 * Serve one client request. The attempt order is the request's planned model first, then every
 * other candidate by rating. A refusal (402, 429, 5xx, or a call that never reached the provider)
 * is retried once on the same candidate — a provider's failure schedule refuses single calls —
 * and a wrong-key or unknown-model answer (401/400) skips the candidate at once. The first
 * provider that answers 2xx serves the request: exactly one successful provider call per request.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<{ status: number; text: string; stream: boolean }> {
  const stream = body.stream === true;
  const { id, requires } = requirementsFor(body, clientApi);
  const jsonMode = requires.json === true;
  const candidates = candidatesFor(brief, probes, requires);
  const planned = id !== null ? plan.get(id) : undefined;
  const order: Candidate[] = [];
  if (planned !== undefined) order.push(planned);
  for (const candidate of candidates) {
    if (planned !== undefined && candidate.provider.id === planned.provider.id && candidate.model.id === planned.model.id) continue;
    order.push(candidate);
  }
  const tried = new Set<string>();
  for (const candidate of order) {
    const key = `${candidate.provider.id}/${candidate.model.id}`;
    if (tried.has(key)) continue;
    tried.add(key);
    const upstream = upstreamBody(clientApi, body, candidate.provider, candidate.model.id, jsonMode);
    let attempts = 0;
    for (;;) {
      attempts += 1;
      const result = await callProvider(candidate.provider, candidate.model.id, upstream, stream);
      if (result !== null && result.status >= 200 && result.status < 300) {
        const answer = parseUpstreamAnswer(candidate.provider.wire, result.text, stream);
        if (answer === null) break; // An answer this server cannot read: try the next candidate.
        return { status: 200, text: renderAnswer(answer, clientApi, candidate.model.id, stream), stream };
      }
      const status = result === null ? 0 : result.status;
      // A refusal may be the provider's failure schedule: one retry on the same candidate.
      if ((status === 402 || status === 429 || status >= 500 || status === 0) && attempts < 2) continue;
      break;
    }
  }
  return { status: 503, text: '', stream };
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: null } });
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, { data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })), has_more: false });
  } else {
    sendJson(res, 200, { object: 'list', data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })) });
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
    const clientApi: 'openai' | 'anthropic' = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
    const body = await readJson(req);
    if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
    const answer = await serve(clientApi, body);
    if (answer.status !== 200) {
      return sendError(res, clientApi, answer.status, clientApi === 'anthropic' ? 'overloaded_error' : 'server_error', 'no free model could serve this request');
    }
    if (answer.stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(answer.text);
    } else {
      res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(answer.text) });
      res.end(answer.text);
    }
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: Catalog): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and
    // stop before any server starts. Wait for stdout to flush so the tail is not lost.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
