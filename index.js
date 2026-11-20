const fs = require('fs');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');
const { Readable } = require('stream');
const { pipeline } = require('stream/promises');
const pino = require('pino');
const {
  default: makeWASocket, useMultiFileAuthState, DisconnectReason,
  fetchLatestBaileysVersion, Browsers,
} = require('@whiskeysockets/baileys');

const MODELS = require('./models.json');
const SYSTEM_PROMPT = fs.readFileSync(path.join(__dirname, 'system_prompt.txt'), 'utf8');
const PORT = process.env.PORT || 10000;
const NUMBER = (process.env.PAIR_NUMBER || '').replace(/\D/g, '');
const HF_TOKEN = process.env.HF_TOKEN || '';
const LLAMA_BIN = process.env.LLAMA_BIN || '/app/llama-server';
const MAX_LOCAL_MB = Number(process.env.MAX_LOCAL_MB || 380); // больше не влезет в 512 МБ ОЗУ Render Free
const FALLBACK_MODEL = process.env.FALLBACK_MODEL || '';
const LLM_URL = (process.env.LLM_URL || '').replace(/\/$/, '');
const MODEL_DIR = '/tmp/models';
const LLAMA_PORT = 8081;
const CTX = Number(process.env.CTX || 2048);
fs.mkdirSync(MODEL_DIR, { recursive: true });

const hfHeaders = HF_TOKEN ? { Authorization: `Bearer ${HF_TOKEN}` } : {};
let pairCode = null, waState = 'starting';
let active = 0, localReady = false, localInfo = 'не используется', apiInfo = 'не проверено', serverState = 'loading';
let llamaProc = null, switchId = 0;
const history = new Map();
const groupAll = new Set(); // группы, где бот отвечает на все сообщения
let localQ = Promise.resolve();

// ---------- бэкенды ----------
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function chatRemote(url, model, messages, maxTokens = 250, timeout = 280000) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...hfHeaders },
    signal: AbortSignal.timeout(timeout),
    body: JSON.stringify({ model, messages, max_tokens: maxTokens, temperature: 0.7 }),
  });
  const raw = await r.text();
  if (!r.ok) throw new Error(`${r.status}: ${raw.slice(0, 250)}`);
  let out;
  try { out = JSON.parse(raw).choices?.[0]?.message?.content?.trim(); }
  catch { throw new Error('сервер вернул не JSON (просыпается?)'); }
  if (!out) throw new Error('пустой ответ');
  return out;
}
const chatSpace = (model, msgs, mt, to) => chatRemote(`${LLM_URL}/v1/chat/completions`, model, msgs, mt, to);
const chatRouter = (model, msgs, mt) => chatRemote('https://router.huggingface.co/v1/chat/completions', model, msgs, mt, 90000);

async function chatLocal(messages) {
  const run = async () => {
    const r = await fetch(`http://127.0.0.1:${LLAMA_PORT}/v1/chat/completions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(240000),
      body: JSON.stringify({ messages, max_tokens: 300, temperature: 0.7 }),
    });
    if (!r.ok) throw new Error('LLM ' + r.status);
    const out = (await r.json()).choices?.[0]?.message?.content?.trim();
    if (!out) throw new Error('пустой ответ');
    return out;
  };
  const p = localQ.then(run, run);
  localQ = p.catch(() => {});
  return p;
}

const soft = msg => { const e = new Error(msg); e.soft = true; return e; };

async function ask(chatId, userText) {
  const h = history.get(chatId) || [];
  h.push({ role: 'user', content: userText });
  while (h.length > 8) h.shift();
  const messages = [{ role: 'system', content: SYSTEM_PROMPT }, ...h];
  const m = MODELS[active];
  let out = null, lastErr = null;
  if (localReady) { try { out = await chatLocal(messages); } catch (e) { lastErr = e; } }
  if (!out && LLM_URL) {
    if (serverState === 'loading') throw soft('Модель загружается на сервере, подожди 2–5 минут ⏳');
    if (serverState === 'error') throw soft('Не удалось загрузить модель: ' + apiInfo);
    try { out = await chatSpace(m.repo, messages); } catch (e) { lastErr = e; }
  }
  if (!out && !LLM_URL) { try { out = await chatRouter(m.repo, messages); } catch (e) { lastErr = e; } }
  if (!out && FALLBACK_MODEL) { try { out = await chatRouter(FALLBACK_MODEL, messages); } catch (e) { lastErr = e; } }
  if (!out) throw lastErr || new Error('нет доступных бэкендов');
  h.push({ role: 'assistant', content: out });
  history.set(chatId, h);
  return out;
}

// ---------- локальный GGUF (в фоне, ничего не блокирует) ----------
async function pickGguf(repo) {
  const r = await fetch(`https://huggingface.co/api/models/${repo}?blobs=true`, { headers: hfHeaders });
  if (!r.ok) throw new Error(`HF ${r.status}`);
  const files = ((await r.json()).siblings || [])
    .filter(s => s.rfilename.toLowerCase().endsWith('.gguf') && !/mmproj/i.test(s.rfilename));
  if (!files.length) throw new Error('в репозитории нет .gguf');
  for (const p of [/q4_k_m/i, /q4_k_s/i, /q4/i, /q5/i, /q3/i, /q2/i, /q8/i]) {
    const f = files.find(x => p.test(x.rfilename)); if (f) return f;
  }
  return files.sort((a, b) => (a.size || 0) - (b.size || 0))[0];
}

async function download(repo, file, dest) {
  if (fs.existsSync(dest)) return;
  const r = await fetch(`https://huggingface.co/${repo}/resolve/main/${file}`, { headers: hfHeaders, redirect: 'follow' });
  if (!r.ok) throw new Error(`скачивание ${r.status}`);
  await pipeline(Readable.fromWeb(r.body), fs.createWriteStream(dest + '.part'));
  fs.renameSync(dest + '.part', dest);
}

async function waitHealth(id, ms = 300000) {
  const end = Date.now() + ms;
  while (Date.now() < end && id === switchId) {
    try { if ((await fetch(`http://127.0.0.1:${LLAMA_PORT}/health`)).ok) return true; } catch {}
    await new Promise(r => setTimeout(r, 2000));
  }
  return false;
}

async function tryLocal(idx, id) {
  localInfo = 'проверяю GGUF...';
  try {
    const m = MODELS[idx];
    const f = await pickGguf(m.repo);
    const mb = (f.size || 0) / 1048576;
    if (mb > MAX_LOCAL_MB) { localInfo = `GGUF ${Math.round(mb)} МБ — не влезает в память, работаю через API`; return; }
    localInfo = `скачиваю GGUF (${Math.round(mb)} МБ)...`;
    for (const x of fs.readdirSync(MODEL_DIR)) fs.rmSync(path.join(MODEL_DIR, x), { force: true });
    const dest = path.join(MODEL_DIR, `${m.id}.gguf`);
    await download(m.repo, f.rfilename, dest);
    if (id !== switchId) return;
    llamaProc = spawn(LLAMA_BIN, ['-m', dest, '--host', '127.0.0.1', '--port', String(LLAMA_PORT),
      '-c', String(CTX), '-t', '2', '-np', '1', '--no-webui'], { stdio: 'inherit' });
    llamaProc.on('exit', () => { llamaProc = null; if (id === switchId) { localReady = false; localInfo = 'остановлена'; } });
    if (await waitHealth(id) && id === switchId) { localReady = true; localInfo = 'локально ✅'; }
    else if (id === switchId) localInfo = 'не запустилась, работаю через API';
  } catch (e) { if (id === switchId) localInfo = `${e.message}, работаю через API`; }
}

async function warmup(idx, id) {
  if (!LLM_URL) { serverState = 'ready'; apiInfo = 'HF Inference API'; return; }
  serverState = 'loading'; apiInfo = 'загружаю модель на сервер (2–5 мин)...';
  for (let i = 0; i < 40 && id === switchId; i++) {
    try {
      await chatSpace(MODELS[idx].repo, [{ role: 'user', content: 'hi' }], 1, 900000);
      if (id === switchId) { serverState = 'ready'; apiInfo = 'сервер HF Space ✅'; }
      return;
    } catch (e) {
      console.log('warmup:', e.message);
      if (/LOAD_FAILED/.test(e.message)) {
        if (id === switchId) { serverState = 'error'; apiInfo = e.message.replace(/^.*LOAD_FAILED:?/, '').slice(0, 220); }
        return;
      }
      await sleep(15000);
    }
  }
  if (id === switchId) { serverState = 'error'; apiInfo = 'сервер не отвечает'; }
}

function activate(idx) {
  switchId++; const id = switchId;
  if (llamaProc) { try { llamaProc.kill('SIGKILL'); } catch {} llamaProc = null; }
  active = idx; localReady = false;
  history.clear();
  warmup(idx, id);
  if (!LLM_URL) tryLocal(idx, id);
}

// ---------- WhatsApp ----------
function getText(m) {
  const x = m.message; if (!x) return '';
  const msg = x.ephemeralMessage?.message || x.viewOnceMessage?.message || x;
  return msg.conversation || msg.extendedTextMessage?.text || msg.imageMessage?.caption || msg.videoMessage?.caption || '';
}
const num = j => (j || '').split('@')[0].split(':')[0];

async function startWA() {
  const { state, saveCreds } = await useMultiFileAuthState('./auth');
  const { version } = await fetchLatestBaileysVersion();
  const sock = makeWASocket({ version, auth: state, logger: pino({ level: 'silent' }),
    browser: Browsers.ubuntu('Chrome'), markOnlineOnConnect: false, syncFullHistory: false });
  sock.ev.on('creds.update', saveCreds);

  if (!sock.authState.creds.registered && NUMBER) {
    setTimeout(async () => {
      try { pairCode = await sock.requestPairingCode(NUMBER); console.log('PAIRING CODE:', pairCode); }
      catch (e) { console.error('pair error', e.message); }
    }, 4000);
  }

  sock.ev.on('connection.update', ({ connection, lastDisconnect }) => {
    if (connection === 'open') { waState = 'connected'; pairCode = null; console.log('WhatsApp подключён'); }
    if (connection === 'close') {
      waState = 'disconnected';
      if (lastDisconnect?.error?.output?.statusCode === DisconnectReason.loggedOut)
        fs.rmSync('./auth', { recursive: true, force: true });
      setTimeout(startWA, 3000);
    }
  });

  sock.ev.on('messages.upsert', ({ messages, type }) => {
    if (type !== 'notify') return;
    for (const m of messages) handle(sock, m).catch(e => console.error('handle', e));
  });
}

async function handle(sock, m) {
  if (!m.message || m.key.fromMe || m.key.remoteJid === 'status@broadcast') return;
  const ts = Number(m.messageTimestamp?.low ?? m.messageTimestamp ?? 0);
  if (ts && Date.now() / 1000 - ts > 90) return; // старые сообщения после переподключения
  const jid = m.key.remoteJid;
  const isGroup = jid.endsWith('@g.us');
  let text = getText(m).trim();
  if (!text) return;

  const me = [num(sock.user?.id), num(sock.user?.lid), num(sock.authState?.creds?.me?.id), num(sock.authState?.creds?.me?.lid)].filter(Boolean);
  const reply = t => sock.sendMessage(jid, { text: t }, { quoted: m });

  let addressed = !isGroup;
  if (isGroup) {
    const ctx = m.message.extendedTextMessage?.contextInfo || {};
    const mentioned = (ctx.mentionedJid || []).some(j => me.includes(num(j))) || me.some(n => text.includes('@' + n));
    const replied = me.includes(num(ctx.participant));
    const byName = /(^|[\s,.!?])(бот|nova|нова|spark|спарк)([\s,.!?:]|$)/i.test(text);
    addressed = mentioned || replied || byName || text.startsWith('/') || groupAll.has(jid);
    if (!addressed) return;
  }
  for (const n of me) text = text.replace(new RegExp('@' + n, 'g'), '');
  text = text.replace(/^(\/ai|бот|nova|нова)[\s,:]+/i, '').trim();
  if (!text) return;

  if (/^\/(models?|модели)(\s|$)/i.test(text)) {
    const arg = text.split(/\s+/)[1];
    if (!arg) {
      return reply('Модели:\n' + MODELS.map((x, i) => `${i + 1}. ${x.name}${i === active ? ' ✅' : ''}`).join('\n') +
        `\n\nСейчас: ${apiInfo}\nЛокально: ${localInfo}\n\nПереключить: /models 2`);
    }
    const i = Number(arg) - 1;
    if (!(i >= 0 && i < MODELS.length)) return reply('Нет такой модели. Напиши /models');
    activate(i);
    await reply(`Переключил на ${MODELS[i].name}. Загружаю модель на сервер — первый ответ будет через 2–5 минут, статус в /models ⏳`);
    return;
  }
  if (/^\/(all|все)(\s|$)/i.test(text)) {
    if (!isGroup) return reply('Эта команда для групп.');
    const on = /on|вкл|да/i.test(text.split(/\s+/)[1] || '');
    on ? groupAll.add(jid) : groupAll.delete(jid);
    return reply(on ? 'Теперь отвечаю на все сообщения в группе.' : 'Отвечаю только когда зовут (@, ответ, "бот,").');
  }
  if (/^\/(reset|сброс)/i.test(text)) { history.delete(jid); return reply('Память чата очищена.'); }
  if (/^\/(help|помощь|start)/i.test(text)) {
    return reply('/models — выбор модели\n/reset — очистить память\n/all on|off — (группа) отвечать на все сообщения\nВ группе зови: @упоминание, ответ на моё сообщение или "бот, ...".');
  }

  await sock.sendPresenceUpdate('composing', jid).catch(() => {});
  try {
    const who = isGroup ? `${m.pushName || 'Участник'}: ` : '';
    await reply(await ask(jid, who + text));
  } catch (e) { await reply(e.soft ? e.message : 'Ошибка: ' + e.message); }
  await sock.sendPresenceUpdate('paused', jid).catch(() => {});
}

// ---------- HTTP (для Render) ----------
http.createServer((req, res) => {
  if (req.url === '/health') { res.end('ok'); return; }
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.end(`<meta name=viewport content="width=device-width"><h2>NOVA WhatsApp Bot</h2>
<p>WhatsApp: ${waState}</p><p>Модель: ${MODELS[active].name}</p><p>${apiInfo}</p><p>Локально: ${localInfo}</p>
${pairCode ? `<h1>Код привязки: ${pairCode}</h1><p>WhatsApp → Связанные устройства → Привязать по номеру телефона</p>` : ''}`);
}).listen(PORT, () => console.log('HTTP on', PORT));

if (process.env.RENDER_EXTERNAL_URL)
  setInterval(() => fetch(process.env.RENDER_EXTERNAL_URL + '/health').catch(() => {}), 10 * 60 * 1000);

if (LLM_URL) setInterval(() => fetch(LLM_URL + '/health', { headers: hfHeaders }).catch(() => {}), 25 * 60 * 1000);

activate(0);
startWA();
