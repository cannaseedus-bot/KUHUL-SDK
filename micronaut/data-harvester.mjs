// data-harvester.mjs — ASX Prime OS autonomous internet data harvesting kernel process
// Port 25109 | Registers with Coordinator | Rate-limited respectful crawling

import http from 'node:http';
import { EventEmitter } from 'node:events';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = 25301;
const COORDINATOR = 'http://127.0.0.1:25100';
const CYCLE_INTERVAL_MS = 300_000;   // 5 minutes between full harvests

// Priority sources — coder-first ordering (MM-CODER training data)
const SOURCES = [
  { id: 'github',        url: 'https://api.github.com/search/repositories?q=language:python&sort=stars&per_page=10', type: 'api' },
  { id: 'stackoverflow', url: 'https://api.stackexchange.com/2.3/questions?order=desc&sort=activity&site=stackoverflow&filter=withbody&pagesize=5', type: 'api' },
  { id: 'arxiv_cs',      url: 'https://export.arxiv.org/api/query?search_query=cat:cs.LG&max_results=5&sortBy=submittedDate', type: 'api' },
  { id: 'huggingface',   url: 'https://huggingface.co/api/models?limit=10&sort=downloads', type: 'api' },
  { id: 'wikipedia',     url: 'https://en.wikipedia.org/api/rest_v1/page/random/summary', type: 'api' },
  { id: 'hackernews',    url: 'https://hacker-news.firebaseio.com/v0/topstories.json', type: 'api' },
  { id: 'openlibrary',   url: 'https://openlibrary.org/search.json?q=computer+science&limit=5', type: 'api' },
  { id: 'nasa',          url: 'https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY', type: 'api' },
];

// Token-bucket rate limiter — 1 request per domain per second
class RateLimiter {
  constructor() { this._last = new Map(); }
  async wait(domain) {
    const gap = 1000 - (Date.now() - (this._last.get(domain) ?? 0));
    if (gap > 0) await new Promise(r => setTimeout(r, gap));
    this._last.set(domain, Date.now());
  }
}

class DataHarvester extends EventEmitter {
  constructor() {
    super();
    this.limiter = new RateLimiter();
    this.cache = [];
    this.cycles = 0;
    this.running = false;
    this.errors = 0;
    this.outDir = path.join(__dirname, '..', 'data', 'harvested');
  }

  async fetchSource(src) {
    const domain = new URL(src.url).hostname;
    await this.limiter.wait(domain);
    const res = await fetch(src.url, {
      headers: { 'User-Agent': 'KUHUL-DataHarvester/1.0 (+respectful-crawling; MM-CODER-training)' },
      signal: AbortSignal.timeout(8000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    return {
      source_id:  src.id,
      source_url: src.url,
      fold:       '⟁COMPUTE_FOLD⟁',
      phase:      "Ch'en",
      timestamp:  new Date().toISOString(),
      raw:        text.slice(0, 8192),
    };
  }

  async harvestCycle() {
    console.log(`[HARVESTER] ⟁COMPUTE_FOLD⟁ cycle ${this.cycles + 1} starting`);
    for (const src of SOURCES) {
      try {
        const record = await this.fetchSource(src);
        this.cache.push(record);
        this.emit('data', record);
        console.log(`[HARVESTER] Sek ${src.id} → ${record.raw.length}b`);
      } catch (e) {
        this.errors++;
        console.error(`[HARVESTER] Xul ${src.id}: ${e.message}`);
      }
    }
    this.cycles++;
    await this.flush();
    this.emit('cycle_complete', { cycle: this.cycles, errors: this.errors });
  }

  async flush() {
    if (!this.cache.length) return;
    await fs.mkdir(this.outDir, { recursive: true });
    const file = path.join(this.outDir, `batch_${Date.now()}.jsonl`);
    const lines = this.cache.splice(0).map(d => JSON.stringify(d)).join('\n');
    await fs.writeFile(file, lines + '\n', 'utf8');
    console.log(`[HARVESTER] Wo → ${path.basename(file)}`);
    return file;
  }

  async start() {
    this.running = true;
    console.log('[HARVESTER] K\'ayab\' learning_lifetime — autonomous harvest active');
    while (this.running) {
      await this.harvestCycle();
      if (this.running) await new Promise(r => setTimeout(r, CYCLE_INTERVAL_MS));
    }
  }

  stop() {
    this.running = false;
    console.log('[HARVESTER] Kumk\'u — harvest stopped');
  }

  status() {
    return {
      running:    this.running,
      cycles:     this.cycles,
      errors:     this.errors,
      cache_size: this.cache.length,
      out_dir:    this.outDir,
    };
  }
}

// ── HTTP API ─────────────────────────────────────────────────────────────────

const harvester = new DataHarvester();

function json(res, code, body) {
  res.writeHead(code, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
}

const server = http.createServer(async (req, res) => {
  const { method, url } = req;

  if (method === 'GET'  && url === '/health')
    return json(res, 200, { ok: true });

  if (method === 'GET'  && url === '/status')
    return json(res, 200, { service: 'data-harvester', port: PORT, ...harvester.status() });

  if (method === 'POST' && url === '/harvest') {
    harvester.harvestCycle().catch(console.error);
    return json(res, 202, { accepted: true, cycle: harvester.cycles + 1 });
  }

  if (method === 'POST' && url === '/stop') {
    harvester.stop();
    return json(res, 200, { stopped: true });
  }

  json(res, 404, { error: 'not found' });
});

async function register() {
  try {
    await fetch(`${COORDINATOR}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: 'data-harvester', port: PORT, type: 'kernel_process', fold: '⟁COMPUTE_FOLD⟁' }),
    });
    console.log('[HARVESTER] Registered with coordinator');
  } catch {
    console.warn('[HARVESTER] Coordinator offline — standalone mode');
  }
}

server.listen(PORT, '127.0.0.1', async () => {
  console.log(`[HARVESTER] Pop harvest_knowledge — port ${PORT}`);
  await register();
  harvester.start().catch(console.error);
});
