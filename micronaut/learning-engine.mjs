// learning-engine.mjs — ASX Prime OS continuous training pipeline connector
// Port 25110 | Hot-swap model improvements | Coordinator-registered

import http from 'node:http';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = 25121;
const COORDINATOR = 'http://127.0.0.1:25100';
const HARVESTER   = 'http://127.0.0.1:25120';

const TRAINER_SCRIPT = path.join(
  __dirname, '..', 'releases', 'v3.5.0-WebX',
  'tools', 'trainers', 'internet_harvester.py'
);

class LearningEngine {
  constructor() {
    this.cycles    = 0;
    this.active    = false;
    this.lastModel = null;
    this.log       = [];
  }

  trainOnBatch(batchDir, modelOut) {
    return new Promise((resolve, reject) => {
      this.active = true;
      const args = ['--batch-dir', batchDir];
      if (modelOut) args.push('--model-out', modelOut);

      const py = spawn('python', [TRAINER_SCRIPT, ...args], {
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      let stdout = '';
      py.stdout.on('data', d => { stdout += d; });
      py.stderr.on('data', d => console.error('[LEARNER]', d.toString().trim()));

      py.on('close', code => {
        this.active = false;
        this.cycles++;
        if (code === 0) {
          try {
            const last = stdout.trim().split('\n').pop();
            resolve(JSON.parse(last));
          } catch {
            resolve({ cycles: this.cycles });
          }
        } else {
          reject(new Error(`trainer exited ${code}`));
        }
      });
    });
  }

  async notifyModelUpdate(modelPath) {
    this.lastModel = modelPath;
    this.log.push({ ts: new Date().toISOString(), model: modelPath, cycle: this.cycles });
    if (this.log.length > 50) this.log.shift();
    console.log(`[LEARNER] Sek hot_swap_active → ${path.basename(modelPath)}`);
    try {
      await fetch(`${COORDINATOR}/notify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          event:      'model_updated',
          source:     'learning-engine',
          model_path: modelPath,
          cycle:      this.cycles,
          fold:       '⟁COMPUTE_FOLD⟁',
        }),
      });
    } catch { /* coordinator may be offline */ }
  }

  status() {
    return {
      active:     this.active,
      cycles:     this.cycles,
      last_model: this.lastModel,
      log_tail:   this.log.slice(-5),
    };
  }
}

// ── HTTP API ─────────────────────────────────────────────────────────────────

const engine = new LearningEngine();

function json(res, code, body) {
  res.writeHead(code, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
}

function readBody(req) {
  return new Promise(resolve => {
    let buf = '';
    req.on('data', c => { buf += c; });
    req.on('end', () => { try { resolve(JSON.parse(buf)); } catch { resolve({}); } });
  });
}

const server = http.createServer(async (req, res) => {
  const { method, url } = req;

  if (method === 'GET'  && url === '/health')
    return json(res, 200, { ok: true });

  if (method === 'GET'  && url === '/status')
    return json(res, 200, { service: 'learning-engine', port: PORT, ...engine.status() });

  if (method === 'POST' && url === '/train') {
    const { batch_dir, model_out } = await readBody(req);
    if (!batch_dir) return json(res, 400, { error: 'batch_dir required' });
    if (engine.active)  return json(res, 409, { error: 'training already active' });

    engine.trainOnBatch(batch_dir, model_out)
      .then(result => engine.notifyModelUpdate(model_out ?? result?.model_path ?? 'E:\\models\\GPT2\\med-GPT'))
      .catch(e => console.error('[LEARNER] train failed:', e.message));

    return json(res, 202, { accepted: true, cycle: engine.cycles + 1 });
  }

  json(res, 404, { error: 'not found' });
});

async function register() {
  try {
    await fetch(`${COORDINATOR}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: 'learning-engine', port: PORT, type: 'kernel_process', fold: '⟁COMPUTE_FOLD⟁' }),
    });
    console.log('[LEARNER] Registered with coordinator');
  } catch {
    console.warn('[LEARNER] Coordinator offline — standalone mode');
  }
}

server.listen(PORT, '127.0.0.1', async () => {
  console.log(`[LEARNER] Pop train_on_new_data — port ${PORT}`);
  await register();
});
