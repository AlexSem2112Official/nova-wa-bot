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
const MODEL_DIR = '/tmp/models';
const LLAMA_PORT = 8081;
const CTX = Number(process.env.CTX || 2048);
fs.mkdirSync(MODEL_DIR, { recursive: true });

let pairCode = null, waState = 'starting', llamaProc = null;
let current = 0, modelState = 'loading', modelError = '';
const history = new Map();
let queue = Promise.resolve();
const hfHeaders = HF_TOKEN ? { Authorization: `Bearer ${HF_TOKEN}` } : {};

// ---------- модели ----------
async function pickGguf(repo) {
  const r = await fetch(`https://huggingface.co/api/models/${repo}`, { headers: hfHeaders });
  if (!r.ok) throw new Error(`HF ${repo}: ${r.status}`);
  const files = ((await r.json()).siblings || []).map(s => s.rfilename)
    .filter(f => f.toLowerCase().endsWith('.gguf') && !/mmproj/i.test(f));
  if (!files.length) throw new Error(`В ${repo} нет .gguf файлов`);
  const pref = [/q4_k_m/i, /q4_k_s/i, /q4/i, /q5/i, /q3/i, /q8/i];
  for (const p of pref) { const f = files.find(x => p.test(x)); if (f) return f; }
  return files[0];
}

async function download(repo, file, dest) {
  if (fs.existsSync(dest)) return;
  console.log('Скачиваю', repo, file);
  const r = await fetch(`https://huggingface.co/${repo}/resolve/main/${file}`, { headers: hfHeaders, redirect: 'follow' });
  if (!r.ok) throw new Error(`Скачивание: ${r.status}`);
  await pipeline(Readable.fromWeb(r.body), fs.createWriteStream(dest + '.part'));
  fs.renameSync(dest + '.part', dest);
}

async function waitHealth(ms = 600000) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    try { const r = await fetch(`http://127.0.0.1:${LLAMA_PORT}/health`); if (r.ok) return; } catch {}
    await new Promise(r => setTimeout(r, 2000));
  }
  throw new Error('llama-server не запустился');
}

async function loadModel(idx) {
  modelState = 'loading'; modelError = '';
  try {
    if (llamaProc) { llamaProc.kill('SIGKILL'); llamaProc = null; await new Promise(r => setTimeout(r, 1000)); }
    const m = MODELS[idx];
    const file = await pickGguf(m.repo);
    const dest = path.join(MODEL_DIR, `${m.id}.gguf`);
    for (const f of fs.readdirSync(MODEL_DIR)) if (f.endsWith('.gguf') && f !== `${m.id}.gguf`) fs.unlinkSync(path.join(MODEL_DIR, f)); // освобождаем диск
    await download(m.repo, file, dest);
    llamaProc = spawn(LLAMA_BIN, ['-m', dest, '--host', '127.0.0.1', '--port', String(LLAMA_PORT),
      '-c', String(CTX), '-t', '2', '-np', '1', '--no-webui'], { stdio: 'inherit' });
    llamaProc.on('exit', () => { llamaProc = null; });
    await waitHealth();
    current = idx; modelState = 'ready';
    console.log('Модель готова:', m.name);
  } catch (e) { modelState = 'error'; modelError = e.message; console.error(e); }
}

async function ask(chatId, userText) {
  const h = history.get(chatId) || [];
  h.push({ role: 'user', content: userText });
  while (h.length > 8) h.shift();
  const r = await fetch(`http://127.0.0.1:${LLAMA_PORT}/v1/chat/completions`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    signal: AbortSignal.timeout(240000),
    body: JSON.stringify({ messages: [{ role: 'system', content: SYSTEM_PROMPT }, ...h],
      max_tokens: 400, temperature: 0.7 }),
  });
  if (!r.ok) throw new Error('LLM ' + r.status);
  const out = ((await r.json()).choices?.[0]?.message?.content || '').trim() || '...';
  h.push({ role: 'assistant', content: out });
  history.set(chatId, h);
  return out;
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
    browser: Browsers.ubuntu('Chrome'), markOnlineOnConnect: false });
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
      const code = lastDisconnect?.error?.output?.statusCode;
      if (code === DisconnectReason.loggedOut) { fs.rmSync('./auth', { recursive: true, force: true }); }
      setTimeout(startWA, 3000);
    }
  });

  sock.ev.on('messages.upsert', ({ messages, type }) => {
    if (type !== 'notify') return;
    for (const m of messages) queue = queue.then(() => handle(sock, m)).catch(e => console.error(e));
  });
}

async function handle(sock, m) {
  if (!m.message || m.key.fromMe || m.key.remoteJid === 'status@broadcast') return;
  const jid = m.key.remoteJid;
  const isGroup = jid.endsWith('@g.us');
  let text = getText(m).trim();
  if (!text) return;

  const me = [num(sock.user?.id), num(sock.user?.lid)].filter(Boolean);
  let addressed = !isGroup;
  if (isGroup) {
    const ctx = m.message.extendedTextMessage?.contextInfo || {};
    const mentioned = (ctx.mentionedJid || []).some(j => me.includes(num(j)));
    const replied = me.includes(num(ctx.participant));
    const byName = /^(\/ai|бот|nova|нова)[\s,:]/i.test(text + ' ');
    addressed = mentioned || replied || byName || text.startsWith('/');
    if (!addressed) return;
  }
  for (const n of me) text = text.replace(new RegExp('@' + n, 'g'), '');
  text = text.replace(/^(\/ai|бот|nova|нова)[\s,:]+/i, '').trim();
  if (!text) return;

  const reply = t => sock.sendMessage(jid, { text: t }, { quoted: m });

  if (/^\/(models?|модели)(\s|$)/i.test(text)) {
    const arg = text.split(/\s+/)[1];
    if (!arg) {
      return reply('Модели:\n' + MODELS.map((x, i) => `${i + 1}. ${x.name}${i === current ? ' ✅' : ''}`).join('\n') +
        '\n\nПереключить: /models 2');
    }
    const i = Number(arg) - 1;
    if (!(i >= 0 && i < MODELS.length)) return reply('Нет такой модели. Напиши /models');
    if (i === current && modelState === 'ready') return reply('Эта модель уже активна.');
    await reply(`Загружаю ${MODELS[i].name}... это может занять несколько минут.`);
    await loadModel(i);
    return reply(modelState === 'ready' ? `Готово: ${MODELS[i].name} ✅` : `Ошибка: ${modelError}`);
  }
  if (/^\/(reset|сброс)/i.test(text)) { history.delete(jid); return reply('Память чата очищена.'); }
  if (/^\/(help|помощь|start)/i.test(text)) {
    return reply('Команды: /models — выбор модели, /reset — очистить память. В группе зови меня через @упоминание, ответ на моё сообщение или "бот, ...".');
  }
  if (modelState === 'loading') return reply('Модель ещё загружается, подожди пару минут ⏳');
  if (modelState === 'error') return reply('Модель не запустилась: ' + modelError);

  await sock.sendPresenceUpdate('composing', jid);
  try {
    const who = isGroup ? `${m.pushName || 'Участник'}: ` : '';
    await reply(await ask(jid, who + text));
  } catch (e) { await reply('Ошибка генерации: ' + e.message); }
  await sock.sendPresenceUpdate('paused', jid);
}

// ---------- HTTP (для Render) ----------
http.createServer((req, res) => {
  if (req.url === '/health') { res.end('ok'); return; }
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.end(`<meta name=viewport content="width=device-width"><h2>NOVA WhatsApp Bot</h2>
<p>WhatsApp: ${waState}</p><p>Модель: ${MODELS[current].name} (${modelState}${modelError ? ': ' + modelError : ''})</p>
${pairCode ? `<h1>Код привязки: ${pairCode}</h1><p>WhatsApp → Связанные устройства → Привязать по номеру телефона</p>` : ''}`);
}).listen(PORT, () => console.log('HTTP on', PORT));

if (process.env.RENDER_EXTERNAL_URL)
  setInterval(() => fetch(process.env.RENDER_EXTERNAL_URL + '/health').catch(() => {}), 10 * 60 * 1000);

loadModel(0);
startWA();
