/* ===========================================================================
   Praxis — client

   No framework, no build step. One file, plain DOM. The whole app is small
   enough that a framework would cost more than it saves, and this way you can
   read it end to end.
   =========================================================================== */

const state = {
  conversationId: null,
  messages: [],          // {role, content, tools:[], meta}
  mode: 'standard',
  modes: [],
  strategy: 'single',
  strategies: [],
  promises: {},
  council: null,
  tools: [],
  providers: [],
  conversations: [],
  brand: { name: 'Praxis' },
  streaming: false,
  abort: null,
  attachments: [],
  waitTimer: null,
};

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

/* --- markdown -------------------------------------------------------------
   Everything is HTML-escaped before any markup is added, so model output can
   never inject into the page. Small by design: the common 90% of markdown,
   rendered predictably, with no dependency. */

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderMarkdown(src) {
  const blocks = [];
  // Pull fenced code out first so its contents never hit the inline rules.
  // The placeholder gets its own line so the block loop below treats it as a
  // block rather than folding it into a paragraph, and the token is distinctive
  // enough that prose can never collide with it.
  let text = escapeHtml(src).replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    blocks.push('<pre><button class="copy-code">copy</button><code>' +
                code.replace(/\n$/, '') + '</code></pre>');
    return '\n%%PXBLOCK' + (blocks.length - 1) + '%%\n';
  });

  const lines = text.split('\n');
  const out = [];
  let list = null, para = [], inTable = false, tableRows = [];

  const flushPara = () => {
    if (para.length) { out.push('<p>' + inline(para.join(' ')) + '</p>'); para = []; }
  };
  const flushList = () => { if (list) { out.push('</' + list + '>'); list = null; } };
  const flushTable = () => {
    if (!tableRows.length) return;
    const cells = (row, tag) => row.split('|').slice(1, -1)
      .map(c => '<' + tag + '>' + inline(c.trim()) + '</' + tag + '>').join('');
    const head = '<thead><tr>' + cells(tableRows[0], 'th') + '</tr></thead>';
    const body = tableRows.slice(2).map(r => '<tr>' + cells(r, 'td') + '</tr>').join('');
    out.push('<table>' + head + '<tbody>' + body + '</tbody></table>');
    tableRows = []; inTable = false;
  };

  for (const raw of lines) {
    const line = raw.trimEnd();

    const held = line.trim().match(/^%%PXBLOCK(\d+)%%$/);
    if (held) {
      flushPara(); flushList(); flushTable();
      out.push(blocks[Number(held[1])]);
      continue;
    }

    if (inTable) {
      if (/^\s*\|.*\|\s*$/.test(line)) { tableRows.push(line.trim()); continue; }
      flushTable();
    }
    if (/^\s*\|.*\|\s*$/.test(line) && !list) {
      flushPara(); inTable = true; tableRows = [line.trim()]; continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      flushPara(); flushList();
      const level = heading[1].length;
      out.push('<h' + level + '>' + inline(heading[2]) + '</h' + level + '>');
      continue;
    }
    if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) {
      flushPara(); flushList(); out.push('<hr>'); continue;
    }

    const quote = line.match(/^>\s?(.*)$/);
    if (quote) {
      flushPara(); flushList();
      out.push('<blockquote>' + inline(quote[1]) + '</blockquote>');
      continue;
    }

    const ul = line.match(/^\s*[-*+]\s+(.*)$/);
    const ol = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const want = ul ? 'ul' : 'ol';
      if (list !== want) { flushList(); out.push('<' + want + '>'); list = want; }
      out.push('<li>' + inline(ul ? ul[1] : ol[2]) + '</li>');
      continue;
    }

    if (!line.trim()) { flushPara(); flushList(); continue; }
    if (list) flushList();
    para.push(line);
  }
  flushPara(); flushList(); flushTable();

  return out.join('\n').replace(/%%PXBLOCK(\d+)%%/g, (_, i) => blocks[i]);
}

function inline(s) {
  return s
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Bold must run first and must tolerate asterisks inside it — "**1920*1080
    // = 2073600**" is a real thing a model writes, and [^*]+ cannot match it.
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // Italic only when the delimiters hug non-space, so "2 * 3 * 4" stays
    // arithmetic instead of becoming emphasis.
    .replace(/(^|[^*\w])\*([^*\s][^*\n]*?[^*\s]|[^*\s])\*/g, '$1<em>$2</em>')
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g,
             '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/(^|\s)(https?:\/\/[^\s<]+)/g,
             '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');
}

/* --- toasts --------------------------------------------------------------- */

let toastTimer = null;
function toast(text) {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();
  const t = el('div', 'toast', text);
  document.body.appendChild(t);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.remove(), 2400);
}

/* --- boot ----------------------------------------------------------------- */

async function boot() {
  const data = await (await fetch('/api/bootstrap')).json();
  state.brand = data.brand;
  state.modes = data.modes;
  state.tools = data.tools;
  state.providers = data.providers;
  state.conversations = data.conversations;
  state.strategies = data.strategies || [];
  state.council = data.council || null;
  state.mode = localStorage.getItem('praxis.mode') || 'standard';
  state.strategy = localStorage.getItem('praxis.strategy') ||
                   (data.council && data.council.default_strategy) || 'single';

  document.title = data.brand.name;
  $('brandName').textContent = data.brand.name;
  $('brandVer').textContent = 'v' + data.brand.version;
  document.querySelectorAll('.brand-mark').forEach(n => {
    n.textContent = data.brand.name;   // hidden by text-indent; kept for a11y
  });

  const p = data.provider;
  $('providerDot').className = 'dot ' + (p.ok ? 'ok' : 'bad');
  $('providerText').textContent = p.detail;
  $('providerPill').title = p.notes.length
    ? 'Fell back. Skipped:\n' + p.notes.join('\n') : p.detail;

  $('toolChip').textContent = data.features.tools
    ? data.tools.length + ' tools' : 'tools off';
  $('toolChip').title = data.tools.map(t => t.icon + ' ' + t.name).join('\n');

  buildModeMenu();
  applyMode(state.mode);
  buildStrategyMenu();
  applyStrategy(state.strategy);
  renderConversations();
  renderEmpty();

  document.documentElement.dataset.theme =
    localStorage.getItem('praxis.theme') || 'light';
}

/* --- modes ---------------------------------------------------------------- */

function buildModeMenu() {
  const menu = $('modeMenu');
  menu.innerHTML = '';
  for (const m of state.modes) {
    const b = el('button', 'mode-item' + (m.key === state.mode ? ' active' : ''));
    b.innerHTML = '<span class="mode-icon">' + m.icon + '</span>' +
      '<span><span class="mode-item-label">' + escapeHtml(m.label) + '</span><br>' +
      '<span class="mode-item-blurb">' + escapeHtml(m.blurb) + '</span></span>';
    b.onclick = () => { applyMode(m.key); menu.classList.add('hidden'); };
    menu.appendChild(b);
  }
}

function applyMode(key) {
  const m = state.modes.find(x => x.key === key) || state.modes[0];
  if (!m) return;
  state.mode = m.key;
  localStorage.setItem('praxis.mode', m.key);
  $('modeIcon').textContent = m.icon;
  $('modeLabel').textContent = m.label;
  buildModeMenu();
  updatePromiseChip();
}

/* --- strategy (how many models answer) ------------------------------------ */

function buildStrategyMenu() {
  const menu = $('strategyMenu');
  if (!menu) return;
  menu.innerHTML = '';
  for (const s of state.strategies) {
    const b = el('button', 'mode-item' + (s.key === state.strategy ? ' active' : ''));
    let blurb = escapeHtml(s.blurb);
    if (s.key !== 'single' && state.council && state.council.members.length) {
      blurb += '<br><span style="opacity:.7">' +
               escapeHtml(state.council.members.join(' · ')) + '</span>';
    }
    b.innerHTML = '<span class="mode-icon">' + s.icon + '</span>' +
      '<span><span class="mode-item-label">' + escapeHtml(s.label) + '</span><br>' +
      '<span class="mode-item-blurb">' + blurb + '</span></span>';
    b.onclick = () => { applyStrategy(s.key); menu.classList.add('hidden'); };
    menu.appendChild(b);
  }
}

function applyStrategy(key) {
  const s = state.strategies.find(x => x.key === key) || state.strategies[0];
  if (!s) return;
  state.strategy = s.key;
  localStorage.setItem('praxis.strategy', s.key);
  $('strategyIcon').textContent = s.icon;
  $('strategyLabel').textContent = s.label;
  buildStrategyMenu();
}

// Show what this mode has actually committed to, when it has committed to
// anything checkable. A promise nobody can see is a promise nobody can hold
// you to.
function updatePromiseChip() {
  const chip = $('promiseChip');
  if (!chip) return;
  const p = (state.promises && state.promises[state.mode]) || [];
  if (!p.length) { chip.classList.add('hidden'); return; }
  chip.classList.remove('hidden');
  chip.textContent = '✓ ' + p.length + ' checked';
  chip.title = 'This mode is held to:\n· ' + p.join('\n· ') +
               '\nIf the answer breaks one, it is rewritten automatically.';
}

/* --- conversations -------------------------------------------------------- */

function renderConversations(list) {
  const items = list || state.conversations;
  const box = $('convList');
  box.innerHTML = '';
  if (!items.length) {
    box.appendChild(el('div', 'empty-note', 'No conversations yet.'));
    return;
  }
  const groups = [
    ['Pinned', items.filter(c => c.pinned)],
    ['Recent', items.filter(c => !c.pinned)],
  ];
  for (const [name, group] of groups) {
    if (!group.length) continue;
    box.appendChild(el('div', 'conv-group', name));
    for (const c of group) {
      const row = el('div', 'conv' + (c.id === state.conversationId ? ' active' : ''));
      if (c.pinned) row.appendChild(el('span', 'conv-pin', '●'));
      row.appendChild(el('div', 'conv-title', c.title));
      const actions = el('div', 'conv-actions');

      const pin = el('button', null, c.pinned ? '⊘' : '⊙');
      pin.title = c.pinned ? 'Unpin' : 'Pin';
      pin.onclick = async (e) => {
        e.stopPropagation();
        await fetch('/api/conversations/' + c.id, {
          method: 'PATCH', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pinned: !c.pinned }),
        });
        await refreshConversations();
      };

      const del = el('button', null, '×');
      del.title = 'Delete';
      del.onclick = async (e) => {
        e.stopPropagation();
        if (!confirm('Delete "' + c.title + '"?')) return;
        await fetch('/api/conversations/' + c.id, { method: 'DELETE' });
        if (c.id === state.conversationId) newChat();
        await refreshConversations();
      };

      actions.append(pin, del);
      row.appendChild(actions);
      row.onclick = () => openConversation(c.id);
      box.appendChild(row);
    }
  }
}

async function refreshConversations() {
  state.conversations = await (await fetch('/api/conversations')).json();
  renderConversations();
}

async function openConversation(id) {
  const data = await (await fetch('/api/conversations/' + id)).json();
  state.conversationId = id;
  state.mode = data.mode || 'standard';
  applyMode(state.mode);
  state.messages = data.messages.map(m => ({
    role: m.role, content: m.content, id: m.id, tools: [], meta: m.meta,
  }));
  $('convTitle').textContent = data.title;
  renderThread();
  renderConversations();
  if (window.innerWidth < 760) $('sidebar').classList.add('collapsed');
}

function newChat() {
  state.conversationId = null;
  state.messages = [];
  state.attachments = [];
  renderAttachments();
  $('convTitle').textContent = '';
  renderEmpty();
  renderConversations();
  $('input').focus();
}

/* --- thread rendering ------------------------------------------------------ */

const STARTERS = [
  ['founder', 'Pressure-test my startup idea and tell me what would kill it'],
  ['build', 'Design and scaffold a REST API with auth, tests, and deployment'],
  ['teacher', 'Explain how transformers work, starting from what I already know'],
  ['direct', 'Review this plan and tell me only what is wrong with it'],
];

function renderEmpty() {
  const inner = $('threadInner');
  inner.innerHTML = '';
  const wrap = el('div', 'empty');
  wrap.innerHTML =
    '<div class="empty-mark">' + escapeHtml(state.brand.name || 'Praxis') + '</div>' +
    '<h2>What are we building?</h2>' +
    '<div class="empty-tag">' + escapeHtml(state.brand.tagline || '') + '</div>';
  const grid = el('div', 'starters');
  for (const [mode, text] of STARTERS) {
    const m = state.modes.find(x => x.key === mode);
    const b = el('button', 'starter');
    b.innerHTML =
      '<div class="starter-mode">' +
      (m ? m.icon + ' ' + escapeHtml(m.label) : mode) + '</div>' +
      '<div class="starter-text">' + escapeHtml(text) + '</div>';
    b.onclick = () => { applyMode(mode); $('input').value = text; send(); };
    grid.appendChild(b);
  }
  wrap.appendChild(grid);
  inner.appendChild(wrap);
}

function renderThread() {
  updateTokenMeter();
  const inner = $('threadInner');
  inner.innerHTML = '';
  if (!state.messages.length) { renderEmpty(); return; }
  state.messages.forEach((m, i) => inner.appendChild(messageNode(m, i)));
  scrollDown();
}

function messageNode(m, index) {
  const node = el('div', 'msg ' + m.role + (m.error ? ' error' : ''));
  node.dataset.index = index;

  const head = el('div', 'msg-head');
  head.appendChild(el('span', null, m.role === 'user'
    ? (state.brand.owner || 'You') : state.brand.name));
  if (m.meta && m.meta.mode && m.meta.mode !== 'standard') {
    const mode = state.modes.find(x => x.key === m.meta.mode);
    if (mode) head.appendChild(el('span', 'chip', mode.icon + ' ' + mode.label));
  }
  node.appendChild(head);

  if (m.breach) {
    const n = el('div', 'mode-breach-note');
    n.innerHTML = '<span class="ic">↻</span><span>Rewritten — ' +
      escapeHtml(m.breach.mode) + ' mode broke: ' +
      escapeHtml(m.breach.broke.join(', ')) + '</span>';
    node.appendChild(n);
  }
  if (m.council) node.appendChild(councilNode(m.council));
  if (m.cascade) node.appendChild(cascadeNode(m.cascade));

  if (m.tools && m.tools.length) {
    const box = el('div', 'msg-tools');
    m.tools.forEach(t => box.appendChild(toolNode(t)));
    node.appendChild(box);
  }

  const carried = (m.meta && m.meta.attachments) || [];
  if (carried.length) {
    const box = el('div', 'msg-files');
    carried.forEach(f => {
      const chip = el('span', 'msg-file');
      chip.textContent = '▤ ' + (f.filename || 'file') +
        (f.characters ? '  ' + f.characters.toLocaleString() + ' chars' : '');
      box.appendChild(chip);
    });
    node.appendChild(box);
  }

  const body = el('div', 'msg-body' + (m.role === 'assistant' ? ' md' : ''));
  if (m.role === 'assistant') body.innerHTML = renderMarkdown(m.content || '');
  else body.textContent = m.content;
  node.appendChild(body);

  node.appendChild(footer(m, index));
  return node;
}

function toolNode(t) {
  const box = el('div', 'tool-run');
  const head = el('button', 'tool-run-head');
  const status = t.pending
    ? '<span class="tool-spin">◐</span>'
    : '<span class="tool-run-status ' + (t.ok ? 'ok' : 'bad') + '">' +
      (t.ok ? '✓' : '✕') + '</span>';
  const argText = Object.values(t.args || {}).join(', ').slice(0, 70);
  head.innerHTML =
    '<span class="tool-run-icon">' + (t.icon || '▸') + '</span>' +
    '<span class="tool-run-name">' + escapeHtml(t.name) + '</span>' +
    '<span style="color:var(--text-faint);overflow:hidden;text-overflow:ellipsis;' +
    'white-space:nowrap">' + escapeHtml(argText) + '</span>' + status;
  box.appendChild(head);
  if (t.output) {
    const body = el('div', 'tool-run-body', t.output);
    body.classList.add('hidden');
    head.onclick = () => body.classList.toggle('hidden');
    box.appendChild(body);
  }
  return box;
}

function councilNode(c) {
  const box = el('div', 'council');
  const head = el('div', 'council-head');
  head.innerHTML = '<span class="ic">' + (c.strategy === 'race' ? '⚡' : '⚖') + '</span>' +
    '<span class="ti">' + (c.strategy === 'race' ? 'Race' : 'Council') + '</span>' +
    '<span class="sub">' + escapeHtml(c.members.join(' · ')) + '</span>';
  box.appendChild(head);

  const body = el('div', 'council-body');
  for (const m of c.entries) {
    const row = el('div', 'cm' + (m.winner ? ' win' : '') + (m.ok ? '' : ' err'));
    row.innerHTML =
      '<span class="cm-tag">' + escapeHtml(m.label) + '</span>' +
      '<span class="cm-name">' + escapeHtml(m.provider) + '</span>' +
      '<span class="cm-prev">' + escapeHtml(m.error || m.preview || '') + '</span>' +
      (m.elapsed ? '<span class="cm-time">' + m.elapsed + 's</span>' : '') +
      '<span class="cm-mark ' + (m.ok ? 'ok' : 'no') + '">' +
        (m.pending ? '◐' : (m.ok ? '✓' : '✕')) + '</span>';
    body.appendChild(row);
  }
  box.appendChild(body);

  if (c.verdict) {
    const v = el('div', 'council-verdict' + (c.verdict.noConsensus ? ' none' : ''));
    v.innerHTML = c.verdict.noConsensus
      ? '<b>No consensus</b> — ' + escapeHtml(c.verdict.reason) +
        '<span class="how">best of a weak field</span>'
      : 'Winner <b>' + escapeHtml(c.verdict.winner) + '</b> — ' +
        escapeHtml(c.verdict.reason) +
        '<span class="how">' + escapeHtml(c.verdict.method) + '</span>';
    box.appendChild(v);
  }
  for (const note of (c.dropped || [])) {
    const d = el('div', 'council-verdict');
    d.style.color = 'var(--text-faint)';
    d.textContent = 'Skipped — ' + note;
    box.appendChild(d);
  }
  return box;
}

function cascadeNode(info) {
  const n = el('div', 'cascade-note');
  n.innerHTML = '<span class="ic">↗</span><span>Escalated to <b>' +
    escapeHtml(info.to) + '</b> — ' + escapeHtml(info.reason) + '</span>';
  return n;
}

function footer(m, index) {
  const foot = el('div', 'msg-foot');
  const copy = el('button', null, 'Copy');
  copy.onclick = () => {
    navigator.clipboard.writeText(m.content).then(() => toast('Copied'));
  };
  foot.appendChild(copy);

  if (m.role === 'assistant' && m.id) {
    const again = el('button', null, 'Regenerate');
    again.onclick = () => regenerate(index);
    foot.appendChild(again);
  }
  if (m.role === 'user') {
    const edit = el('button', null, 'Edit');
    edit.onclick = () => {
      $('input').value = m.content;
      state.messages = state.messages.slice(0, index);
      renderThread();
      $('input').focus();
    };
    foot.appendChild(edit);
  }
  if (m.meta && m.meta.elapsed) {
    foot.appendChild(el('span', 'msg-meta',
      m.meta.elapsed + 's' +
      (m.meta.tool_rounds ? ' · ' + m.meta.tool_rounds + ' tool' : '')));
  }
  return foot;
}

function scrollDown() {
  const t = $('thread');
  t.scrollTop = t.scrollHeight;
}

/* --- sending -------------------------------------------------------------- */

async function send(overrideText) {
  const input = $('input');
  let text = (overrideText != null ? overrideText : input.value).trim();
  if (state.streaming) return;
  if (!text && !state.attachments.length) return;

  // Attachments travel as ids. Pasting the file's text into the message —
  // which this used to do — meant a PDF turned the user's own bubble into
  // forty pages of PDF. The server folds the text into the prompt at compose
  // time, so the model still reads every word of it.
  const files = state.attachments.map(a => ({
    id: a.id, filename: a.filename, characters: a.characters || 0,
  }));
  if (files.length) {
    state.attachments = [];
    renderAttachments();
  }
  if (!text && !files.length) return;

  input.value = '';
  input.style.height = 'auto';

  state.messages.push({
    role: 'user', content: text, meta: files.length ? { attachments: files } : null,
  });
  renderThread();
  await stream({ message: text, attachments: files });
}

async function regenerate(index) {
  const target = state.messages[index];
  if (!target || !target.id) return;
  state.messages = state.messages.slice(0, index);
  renderThread();
  await stream({ message: '', regenerate_from: target.id });
}

async function stream(payload) {
  state.streaming = true;
  $('sendBtn').disabled = true;
  $('stopBtn').classList.remove('hidden');

  const assistant = {
    role: 'assistant', content: '', tools: [], meta: { mode: state.mode },
  };
  state.messages.push(assistant);
  renderThread();

  const node = $('threadInner').lastChild;
  const body = node.querySelector('.msg-body');
  const toolBox = el('div', 'msg-tools');
  node.insertBefore(toolBox, body);
  body.classList.add('caret');

  const controller = new AbortController();
  state.abort = controller;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(Object.assign({
        conversation_id: state.conversationId,
        mode: state.mode,
        strategy: state.strategy,
      }, payload)),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6);
        if (raw === '[DONE]') continue;
        let evt;
        try { evt = JSON.parse(raw); } catch (err) { continue; }
        handleEvent(evt, assistant, body, toolBox, node);
      }
    }
  } catch (e) {
    if (e.name !== 'AbortError') {
      assistant.error = true;
      assistant.content = String(e);
      body.textContent = assistant.content;
      node.classList.add('error');
    }
  } finally {
    clearInterval(state.waitTimer);
    state.waitTimer = null;
    body.classList.remove('caret');
    state.streaming = false;
    state.abort = null;
    $('sendBtn').disabled = false;
    $('stopBtn').classList.add('hidden');
    renderThread();
    refreshConversations();
    $('input').focus();
  }
}

// Re-render the council / cascade panel in place, above the streaming text.
function renderCouncil(assistant, node) {
  if (!node) return;
  node.querySelectorAll('.council, .cascade-note').forEach(n => n.remove());
  const anchor = node.querySelector('.msg-tools') || node.querySelector('.msg-body');
  if (assistant.council) node.insertBefore(councilNode(assistant.council), anchor);
  if (assistant.cascade) node.insertBefore(cascadeNode(assistant.cascade), anchor);
  scrollDown();
}

function handleEvent(evt, assistant, body, toolBox, node) {
  const d = evt.data || {};
  switch (evt.type) {
    case 'conversation':
      state.conversationId = d.id;
      break;

    case 'start':
      if (d.notes && d.notes.length) {
        toast('Using ' + d.provider_label + ' — ' + d.notes[0]);
      }
      state.promises = state.promises || {};
      state.promises[d.mode] = d.promises || [];
      updatePromiseChip();
      break;

    // The mode broke a promise it made. A rewrite may follow — and if one
    // does, `constraint_retry` clears the draft. Clearing here as well used to
    // wipe the answer on the *last* attempt, where no rewrite follows and the
    // only thing left to render was the failure note. That is the blank bubble.
    case 'mode_breach':
      assistant.breach = d;
      toast(d.mode + ' mode broke its promise — rewriting');
      break;

    case 'token':
      if (state.waitTimer) { clearInterval(state.waitTimer); state.waitTimer = null; }
      assistant.content += d.text;
      body.innerHTML = renderMarkdown(assistant.content);
      body.classList.add('caret');
      scrollDown();
      break;

    case 'tool_call': {
      const t = { name: d.name, args: d.args, icon: d.icon, pending: true };
      assistant.tools.push(t);
      toolBox.appendChild(toolNode(t));
      scrollDown();
      break;
    }

    case 'tool_result': {
      const t = assistant.tools.find(x => x.pending && x.name === d.name);
      if (t) {
        t.pending = false; t.ok = d.ok; t.output = d.output;
        toolBox.innerHTML = '';
        assistant.tools.forEach(x => toolBox.appendChild(toolNode(x)));
      }
      scrollDown();
      break;
    }

    case 'memory':
      toast('Remembered: ' + d.content.slice(0, 52));
      break;

    // A checker found the answer broke a constraint the request stated, and
    // the model is rewriting it. Clear what streamed so far — showing the
    // rejected draft above its replacement would just be confusing.
    case 'council_start': {
      assistant.council = {
        strategy: d.strategy, members: d.members, dropped: d.dropped,
        entries: d.members.map((m, i) => ({
          label: String.fromCharCode(65 + i), provider: m,
          pending: true, ok: false, preview: 'thinking…',
        })),
        verdict: null,
      };
      renderCouncil(assistant, node);
      break;
    }

    case 'council_member': {
      if (!assistant.council) break;
      const e = assistant.council.entries.find(x => x.provider === d.provider)
             || assistant.council.entries.find(x => x.label === d.label);
      if (e) Object.assign(e, {
        pending: false, ok: d.ok, elapsed: d.elapsed,
        error: d.error, preview: d.preview,
      });
      renderCouncil(assistant, node);
      break;
    }

    case 'council_judging':
      if (assistant.council) {
        assistant.council.judging = true;
        toast('Judging with ' + d.judge);
      }
      break;

    case 'council_verdict': {
      if (!assistant.council) break;
      assistant.council.verdict = {
        winner: d.winner, reason: d.reason, method: d.method,
        noConsensus: !!d.no_consensus,
      };
      for (const e of assistant.council.entries) e.winner = (e.label === d.label);
      renderCouncil(assistant, node);
      break;
    }

    case 'cascade_escalate':
      assistant.cascade = { reason: d.reason, from: d.from, to: d.to };
      assistant.content = '';      // the weak draft is being replaced
      body.innerHTML = '';
      renderCouncil(assistant, node);
      toast('Escalating to ' + d.to);
      break;

    case 'constraint_retry':
      assistant.content = '';
      assistant.tools = [];
      toolBox.innerHTML = '';
      body.innerHTML = '';
      body.classList.add('caret');
      toast('Constraint not met — rewriting (attempt ' + d.attempt + ')');
      break;

    case 'constraint_ok':
      toast('Constraint satisfied on attempt ' + d.attempts);
      break;

    case 'constraint_failed':
      toast('Could not meet the constraint after ' + d.attempts + ' attempts');
      break;

    case 'done':
      assistant.id = d.message_id;
      assistant.meta = Object.assign({}, assistant.meta, d);
      break;

    // Rate limited, not broken. The agent is waiting it out, so show the wait
    // counting down in place of the answer rather than leaving a dead pause.
    case 'rate_limited': {
      const until = Date.now() + d.seconds * 1000;
      clearInterval(state.waitTimer);
      const tick = () => {
        const left = Math.max(0, Math.ceil((until - Date.now()) / 1000));
        body.innerHTML = '<p class="waiting">Rate limit reached — retrying in ' +
          left + 's <span class="dim">(attempt ' + d.attempt + ' of ' +
          (d.of + 1) +
          (d.dropped_turns ? ', with ' + d.dropped_turns + ' older turn' +
            (d.dropped_turns > 1 ? 's' : '') + ' dropped to fit' : '') +
          ')</span></p>';
        if (left <= 0) clearInterval(state.waitTimer);
      };
      tick();
      state.waitTimer = setInterval(tick, 500);
      toast('Rate limited — waiting ' + d.seconds + 's');
      break;
    }

    case 'empty_answer':
      assistant.error = true;
      break;

    case 'error':
      clearInterval(state.waitTimer);
      assistant.error = true;
      assistant.content += (assistant.content ? '\n\n' : '') +
                           '**Error:** ' + d.message;
      // Render it now. Waiting for the final re-render left the bubble blank
      // for the whole of a failed turn.
      body.innerHTML = renderMarkdown(assistant.content);
      body.classList.remove('caret');
      if (node) node.classList.add('error');
      scrollDown();
      break;
  }
}

/* --- attachments ---------------------------------------------------------- */

function renderAttachments() {
  const box = $('attachments');
  box.innerHTML = '';
  box.classList.toggle('hidden', !state.attachments.length);
  state.attachments.forEach((a, i) => {
    const chip = el('div', 'attachment');
    chip.appendChild(el('span', null, '▤ ' + a.filename));
    const x = el('button', null, '×');
    x.onclick = () => { state.attachments.splice(i, 1); renderAttachments(); };
    chip.appendChild(x);
    box.appendChild(chip);
  });
}

async function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  const url = '/api/upload' +
    (state.conversationId ? '?conversation_id=' + state.conversationId : '');
  const res = await fetch(url, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    toast(err.detail || 'Upload failed');
    return;
  }
  state.attachments.push(await res.json());
  renderAttachments();
  toast('Attached ' + file.name);
}

/* --- drawers -------------------------------------------------------------- */

function closeDrawers() {
  document.querySelectorAll('.drawer, .scrim, .palette').forEach(n => n.remove());
}

function drawer(title, build, footBuild) {
  closeDrawers();
  const scrim = el('div', 'scrim');
  scrim.onclick = closeDrawers;
  const d = el('div', 'drawer');
  const head = el('div', 'drawer-head');
  head.appendChild(el('h3', null, title));
  head.appendChild(el('div', 'spacer'));
  const close = el('button', 'btn-icon', '×');
  close.onclick = closeDrawers;
  head.appendChild(close);
  const bodyEl = el('div', 'drawer-body');
  d.append(head, bodyEl);
  if (footBuild) {
    const foot = el('div', 'drawer-foot');
    footBuild(foot, bodyEl);
    d.appendChild(foot);
  }
  document.body.append(scrim, d);
  build(bodyEl);
  return bodyEl;
}

async function openMemory() {
  drawer('Memory', async (box) => {
    box.appendChild(el('div', 'empty-note', 'Loading…'));
    const memories = await (await fetch('/api/memories')).json();
    box.innerHTML = '';
    if (!memories.length) {
      box.appendChild(el('div', 'empty-note',
        'Nothing remembered yet.\n\nTell it something worth keeping — a preference, ' +
        'a project you are working on, a decision you made — and it will store it ' +
        'here. Everything is editable and deletable.'));
      return;
    }
    for (const m of memories) {
      const card = el('div', 'mem');
      const top = el('div', 'mem-top');
      top.appendChild(el('span', 'mem-cat', m.category));
      if (m.hits) top.appendChild(el('span', 'mem-hits', 'used ' + m.hits + '×'));
      top.appendChild(el('div', 'spacer'));
      const actions = el('div', 'mem-actions');
      const text = el('div', 'mem-text', m.content);

      const edit = el('button', null, 'edit');
      edit.onclick = () => {
        if (text.contentEditable === 'true') {
          text.contentEditable = 'false';
          edit.textContent = 'edit';
          fetch('/api/memories/' + m.id, {
            method: 'PATCH', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content: text.textContent, category: m.category }),
          }).then(() => toast('Updated'));
        } else {
          text.contentEditable = 'true';
          text.focus();
          edit.textContent = 'save';
        }
      };
      const del = el('button', null, 'delete');
      del.onclick = async () => {
        await fetch('/api/memories/' + m.id, { method: 'DELETE' });
        card.remove();
        toast('Forgotten');
      };
      actions.append(edit, del);
      top.appendChild(actions);
      card.append(top, text);
      box.appendChild(card);
    }
  }, (foot) => {
    const clear = el('button', 'btn', 'Forget everything');
    clear.onclick = async () => {
      if (!confirm('Delete every stored memory? This cannot be undone.')) return;
      const r = await (await fetch('/api/memories', { method: 'DELETE' })).json();
      toast('Deleted ' + r.deleted + ' memories');
      closeDrawers();
    };
    foot.appendChild(clear);
  });
}

async function openSettings() {
  drawer('Settings', async (box) => {
    const themeField = el('div', 'field');
    themeField.innerHTML = '<label>Theme</label>';
    const themeSel = el('select');
    ['light', 'dark'].forEach(t => {
      const o = el('option', null, t[0].toUpperCase() + t.slice(1));
      o.value = t;
      if (document.documentElement.dataset.theme === t) o.selected = true;
      themeSel.appendChild(o);
    });
    themeSel.onchange = () => {
      document.documentElement.dataset.theme = themeSel.value;
      localStorage.setItem('praxis.theme', themeSel.value);
    };
    themeField.appendChild(themeSel);
    box.appendChild(themeField);

    const health = await (await fetch('/api/health')).json();
    const provField = el('div', 'field');
    provField.innerHTML = '<label>Provider</label>';
    const status = el('div', 'chip' + (health.ok ? ' chip-accent' : ''));
    status.textContent = health.label + ' — ' + health.detail;
    provField.appendChild(status);
    provField.appendChild(el('div', 'field-hint',
      'Set PROVIDER in your .env file and restart to change this. ' +
      'Free options: ollama (local), groq, gemini.'));
    box.appendChild(provField);

    const listField = el('div', 'field');
    listField.innerHTML = '<label>Available backends</label>';
    for (const p of state.providers) {
      const row = el('div', 'chip');
      row.style.margin = '0 5px 5px 0';
      row.textContent = p.label + (p.free ? ' · free' : '');
      listField.appendChild(row);
    }
    box.appendChild(listField);

    const toolField = el('div', 'field');
    toolField.innerHTML = '<label>Tools enabled</label>';
    for (const t of state.tools) {
      const row = el('div', 'chip');
      row.style.margin = '0 5px 5px 0';
      row.textContent = t.icon + ' ' + t.name;
      row.title = t.description;
      toolField.appendChild(row);
    }
    toolField.appendChild(el('div', 'field-hint',
      'Code execution is off by default — it runs model-written code on this ' +
      'machine. Enable with ENABLE_CODE_EXEC=true only if you understand that.'));
    box.appendChild(toolField);

    const promptField = el('div', 'field');
    promptField.innerHTML = '<label>System prompt</label>';
    const view = el('button', 'btn', 'Inspect the prompt for this mode');
    view.onclick = async () => {
      const p = await (await fetch('/api/prompt?mode=' + state.mode)).json();
      const pre = el('pre', 'prompt-dump', p.system_prompt);
      view.replaceWith(pre);
      promptField.appendChild(el('div', 'field-hint',
        p.characters.toLocaleString() + ' characters ≈ ' + p.estimated_tokens +
        ' tokens · ' + p.memories_included + ' memories · ' +
        p.tools_included.length + ' tools'));
    };
    promptField.appendChild(view);
    promptField.appendChild(el('div', 'field-hint',
      'This is your AI. You should be able to read exactly what it was told.'));
    box.appendChild(promptField);
  });
}

/* --- command palette ------------------------------------------------------ */

function openPalette() {
  closeDrawers();
  const commands = [
    { icon: '+', label: 'New chat', hint: 'Cmd N', run: newChat },
    { icon: '◆', label: 'Memory', hint: 'Cmd M', run: openMemory },
    { icon: '⚙', label: 'Settings', hint: 'Cmd ,', run: openSettings },
    { icon: '↓', label: 'Export this conversation', hint: '', run: exportChat },
    { icon: '◐', label: 'Toggle theme', hint: '', run: toggleTheme },
    { icon: '☰', label: 'Toggle sidebar', hint: '', run: toggleSidebar },
  ].concat(state.modes.map(m => ({
    icon: m.icon, label: 'Mode: ' + m.label, hint: m.blurb,
    run: () => applyMode(m.key),
  }))).concat(state.strategies.map(s => ({
    icon: s.icon, label: 'Strategy: ' + s.label, hint: s.blurb,
    run: () => applyStrategy(s.key),
  })));

  const scrim = el('div', 'scrim');
  scrim.onclick = closeDrawers;
  const box = el('div', 'palette');
  const input = el('input');
  input.placeholder = 'Type a command…';
  const list = el('div', 'palette-list');
  box.append(input, list);
  document.body.append(scrim, box);

  let filtered = commands, selected = 0;
  const draw = () => {
    list.innerHTML = '';
    filtered.forEach((c, i) => {
      const b = el('button', 'palette-item' + (i === selected ? ' sel' : ''));
      b.innerHTML = '<span class="palette-icon">' + c.icon + '</span>' +
        '<span>' + escapeHtml(c.label) + '</span>' +
        '<span class="palette-hint">' + escapeHtml(c.hint || '') + '</span>';
      b.onclick = () => { closeDrawers(); c.run(); };
      list.appendChild(b);
    });
  };
  draw();

  input.oninput = () => {
    const q = input.value.toLowerCase();
    filtered = commands.filter(c => c.label.toLowerCase().includes(q));
    selected = 0;
    draw();
  };
  input.onkeydown = (e) => {
    if (e.key === 'ArrowDown') {
      selected = Math.min(selected + 1, filtered.length - 1); draw(); e.preventDefault();
    }
    if (e.key === 'ArrowUp') {
      selected = Math.max(selected - 1, 0); draw(); e.preventDefault();
    }
    if (e.key === 'Enter' && filtered[selected]) {
      const c = filtered[selected]; closeDrawers(); c.run();
    }
    if (e.key === 'Escape') closeDrawers();
  };
  input.focus();
}

/* --- slash commands -------------------------------------------------------
   Typing "/" in the composer opens a filtered command list. Faster than the
   palette for things you do mid-sentence, and discoverable — the placeholder
   says so, which the palette never could. */

function slashCommands() {
  const cmds = [
    { cmd: '/new',     desc: 'Start a new chat',        run: () => newChat() },
    { cmd: '/memory',  desc: 'Open memory',             run: () => openMemory() },
    { cmd: '/settings',desc: 'Open settings',           run: () => openSettings() },
    { cmd: '/export',  desc: 'Export this conversation',run: () => exportChat() },
    { cmd: '/theme',   desc: 'Toggle light / dark',     run: () => toggleTheme() },
    { cmd: '/keys',    desc: 'Keyboard shortcuts',      run: () => openShortcuts() },
    { cmd: '/council', desc: 'Which models are live',   run: () => openCouncil() },
    { cmd: '/prompt',  desc: 'Inspect the system prompt', run: () => openSettings() },
  ];
  for (const m of state.modes) {
    cmds.push({ cmd: '/' + m.key, desc: 'Mode: ' + m.label + ' — ' + m.blurb,
                run: () => { applyMode(m.key); toast('Mode: ' + m.label); } });
  }
  for (const s of state.strategies) {
    cmds.push({ cmd: '/' + s.key, desc: 'Strategy: ' + s.label + ' — ' + s.blurb,
                run: () => { applyStrategy(s.key); toast('Strategy: ' + s.label); } });
  }
  return cmds;
}

let slashSel = 0;

function slashState() {
  const value = $('input').value;
  // Only a leading slash opens the menu — a URL mid-sentence must not.
  if (!value.startsWith('/')) return null;
  const typed = value.slice(0, value.indexOf(' ') === -1 ? undefined : value.indexOf(' '));
  if (value.includes(' ')) return null;
  const q = typed.toLowerCase();
  const matches = slashCommands().filter(c => c.cmd.startsWith(q));
  return matches.length ? { typed, matches } : null;
}

function renderSlash() {
  const menu = $('slashMenu');
  const st = slashState();
  if (!st) { menu.classList.add('hidden'); return; }
  if (slashSel >= st.matches.length) slashSel = 0;
  menu.innerHTML = '';
  st.matches.slice(0, 9).forEach((c, i) => {
    const b = el('button', 'slash-item' + (i === slashSel ? ' sel' : ''));
    b.innerHTML = '<span class="slash-cmd">' + escapeHtml(c.cmd) + '</span>' +
                  '<span class="slash-desc">' + escapeHtml(c.desc) + '</span>';
    b.onclick = () => runSlash(c);
    menu.appendChild(b);
  });
  menu.classList.remove('hidden');
}

function runSlash(c) {
  $('input').value = '';
  $('input').style.height = 'auto';
  $('slashMenu').classList.add('hidden');
  slashSel = 0;
  c.run();
  $('input').focus();
}

/* --- token meter ----------------------------------------------------------
   Rough, and labelled rough. ~4 characters per token holds well enough for
   English to be a useful gauge and badly enough that it is never presented as
   exact. It exists so a long conversation degrading is visible, not mysterious. */

function updateTokenMeter() {
  const chars = state.messages.reduce((n, m) => n + (m.content || '').length, 0);
  const meter = $('tokenMeter');
  if (!chars) { meter.classList.add('hidden'); return; }
  const tokens = Math.round(chars / 4);
  meter.classList.remove('hidden');
  meter.textContent = '~' + tokens.toLocaleString() + ' tok';
  meter.title = 'Roughly ' + tokens.toLocaleString() +
    ' tokens of conversation so far (estimated at 4 chars/token). ' +
    'Older turns are trimmed before sending.';
  meter.className = 'chip token-meter' +
    (tokens > 12000 ? ' hot' : tokens > 6000 ? ' warn' : '');
}

/* --- dictation ------------------------------------------------------------
   Web Speech API — no library, no key, no upload where the browser does it
   locally. Unsupported browsers get a clear message rather than a dead button. */

let recognition = null;

function toggleDictation() {
  const Impl = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Impl) {
    toast('Dictation needs Chrome or Edge — this browser has no Speech API');
    return;
  }
  const btn = $('micBtn');
  if (recognition) { recognition.stop(); return; }

  recognition = new Impl();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = navigator.language || 'en-US';

  const before = $('input').value;
  recognition.onresult = (e) => {
    let text = '';
    for (let i = e.resultIndex; i < e.results.length; i++) text += e.results[i][0].transcript;
    $('input').value = (before ? before + ' ' : '') + text;
    $('input').style.height = 'auto';
    $('input').style.height = Math.min($('input').scrollHeight, 220) + 'px';
  };
  recognition.onerror = (e) => {
    toast('Dictation error: ' + e.error);
    btn.classList.remove('listening');
    recognition = null;
  };
  recognition.onend = () => { btn.classList.remove('listening'); recognition = null; };
  recognition.start();
  btn.classList.add('listening');
  toast('Listening — click the mic again to stop');
}

/* --- shortcuts sheet ------------------------------------------------------ */

function openShortcuts() {
  const groups = [
    ['Conversation', [
      ['Send', 'Enter'], ['New line', 'Shift Enter'], ['New chat', '⌘ N'],
      ['Stop generating', 'Esc'],
    ]],
    ['Navigate', [
      ['Command palette', '⌘ K'], ['Slash commands', '/'],
      ['Toggle sidebar', '⌘ \\'], ['Shortcuts (this)', '?'],
    ]],
    ['Configure', [
      ['Mode picker', 'click ◈'], ['Strategy picker', '⌘ J'],
      ['Memory', '⌘ M'], ['Settings', '⌘ ,'], ['Dictate', '⌘ ⇧ V'],
    ]],
  ];
  drawer('Keyboard shortcuts', (box) => {
    const wrap = el('div', 'keys');
    for (const [title, rows] of groups) {
      const g = el('div', 'keys-group');
      g.appendChild(el('h4', null, title));
      for (const [what, key] of rows) {
        const r = el('div', 'key-row');
        r.appendChild(el('span', 'what', what));
        const k = el('kbd', null, key);
        r.appendChild(k);
        g.appendChild(r);
      }
      wrap.appendChild(g);
    }
    box.appendChild(wrap);
    box.appendChild(el('div', 'field-hint',
      'On Windows and Linux, ⌘ means Ctrl.'));
  });
}

/* --- council status ------------------------------------------------------- */

async function openCouncil() {
  drawer('Model council', async (box) => {
    box.appendChild(el('div', 'empty-note', 'Checking members…'));
    let data;
    try { data = await (await fetch('/api/council')).json(); }
    catch (e) { box.innerHTML = ''; box.appendChild(el('div','empty-note','Could not reach the server.')); return; }
    box.innerHTML = '';

    const strat = el('div', 'field');
    strat.innerHTML = '<label>Active strategy</label>';
    const cur = state.strategies.find(s => s.key === state.strategy);
    const chip = el('div', 'chip chip-accent');
    chip.textContent = (cur ? cur.icon + ' ' + cur.label : state.strategy);
    strat.appendChild(chip);
    strat.appendChild(el('div', 'field-hint', cur ? cur.blurb : ''));
    box.appendChild(strat);

    const live = el('div', 'field');
    live.innerHTML = '<label>Members reachable now</label>';
    if (!data.live.length) {
      live.appendChild(el('div', 'empty-note',
        'No members are reachable. Council and Race need at least one; ' +
        'add a free Groq or Gemini key, or start Ollama.'));
    }
    for (const m of data.live) {
      const row = el('div', 'chip chip-accent');
      row.style.margin = '0 5px 5px 0';
      row.textContent = '● ' + m;
      live.appendChild(row);
    }
    box.appendChild(live);

    if (data.dropped.length) {
      const bad = el('div', 'field');
      bad.innerHTML = '<label>Unavailable</label>';
      for (const d of data.dropped) {
        const row = el('div', 'chip');
        row.style.cssText = 'margin:0 5px 5px 0;white-space:normal;text-align:left';
        row.textContent = '○ ' + d;
        bad.appendChild(row);
      }
      bad.appendChild(el('div', 'field-hint',
        'Unreachable members are skipped automatically — they never fail a turn.'));
      box.appendChild(bad);
    }

    const judge = el('div', 'field');
    judge.innerHTML = '<label>Judge</label>';
    const jc = el('div', 'chip'); jc.textContent = data.judge;
    judge.appendChild(jc);
    judge.appendChild(el('div', 'field-hint',
      'The judge sees answers labelled A, B, C with no model names, so it ' +
      'cannot favour a backend it recognises. With no judge configured, a ' +
      'stated heuristic decides and says so.'));
    box.appendChild(judge);
  });
}

/* --- misc actions --------------------------------------------------------- */

function toggleTheme() {
  const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('praxis.theme', next);
}

function toggleSidebar() { $('sidebar').classList.toggle('collapsed'); }

function exportChat() {
  if (!state.conversationId) { toast('Nothing to export yet'); return; }
  window.location = '/api/conversations/' + state.conversationId + '/export';
}

/* --- wiring --------------------------------------------------------------- */

$('sendBtn').onclick = () => send();
$('newChat').onclick = newChat;
$('toggleSidebar').onclick = toggleSidebar;
$('toggleTheme').onclick = toggleTheme;
$('openMemory').onclick = openMemory;
$('openSettings').onclick = openSettings;
$('openPalette').onclick = openPalette;
$('exportChat').onclick = exportChat;
$('providerPill').onclick = openSettings;
$('micBtn').onclick = toggleDictation;
$('helpBtn').onclick = openShortcuts;
$('toolChip').onclick = openSettings;
$('stopBtn').onclick = () => { if (state.abort) state.abort.abort(); };
$('attachBtn').onclick = () => $('fileInput').click();
$('fileInput').onchange = (e) => {
  Array.from(e.target.files).forEach(uploadFile);
  e.target.value = '';
};

$('modeBtn').onclick = (e) => {
  e.stopPropagation();
  $('modeMenu').classList.toggle('hidden');
};
$('strategyBtn').onclick = (e) => {
  e.stopPropagation();
  $('modeMenu').classList.add('hidden');
  $('strategyMenu').classList.toggle('hidden');
};
$('strategyMenu').onclick = (e) => e.stopPropagation();

document.addEventListener('click', () => {
  $('modeMenu').classList.add('hidden');
  $('strategyMenu').classList.add('hidden');
});
$('modeMenu').onclick = (e) => e.stopPropagation();

const inputEl = $('input');
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 220) + 'px';
  renderSlash();
});
inputEl.addEventListener('keydown', (e) => {
  const st = slashState();
  if (st && !$('slashMenu').classList.contains('hidden')) {
    const n = Math.min(st.matches.length, 9);
    if (e.key === 'ArrowDown') { slashSel = (slashSel + 1) % n; renderSlash(); e.preventDefault(); return; }
    if (e.key === 'ArrowUp')   { slashSel = (slashSel - 1 + n) % n; renderSlash(); e.preventDefault(); return; }
    if (e.key === 'Tab' || (e.key === 'Enter' && !e.shiftKey)) {
      e.preventDefault(); runSlash(st.matches[slashSel]); return;
    }
    if (e.key === 'Escape') { $('slashMenu').classList.add('hidden'); e.preventDefault(); return; }
  }
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

$('search').addEventListener('input', async (e) => {
  const q = e.target.value.trim();
  const url = q ? '/api/conversations?q=' + encodeURIComponent(q) : '/api/conversations';
  renderConversations(await (await fetch(url)).json());
});

document.addEventListener('keydown', (e) => {
  const meta = e.metaKey || e.ctrlKey;
  if (meta && e.key === 'k') { e.preventDefault(); openPalette(); }
  if (meta && e.key === 'n') { e.preventDefault(); newChat(); }
  if (meta && e.key === 'm') { e.preventDefault(); openMemory(); }
  if (meta && e.key === ',') { e.preventDefault(); openSettings(); }
  if (meta && e.key === '\\') { e.preventDefault(); toggleSidebar(); }
  if (meta && e.key === 'j') {
    e.preventDefault();
    $('strategyMenu').classList.toggle('hidden');
  }
  if (meta && e.shiftKey && (e.key === 'V' || e.key === 'v')) {
    e.preventDefault(); toggleDictation();
  }
  // "?" opens shortcuts, but never while typing into a field.
  if (e.key === '?' && !meta &&
      !['INPUT', 'TEXTAREA'].includes((e.target.tagName || '')) &&
      !e.target.isContentEditable) {
    e.preventDefault(); openShortcuts();
  }
  // Esc stops a running generation before it closes anything.
  if (e.key === 'Escape' && state.streaming && state.abort) {
    state.abort.abort();
  }
  if (e.key === 'Escape') {
    closeDrawers();
    $('modeMenu').classList.add('hidden');
    $('strategyMenu').classList.add('hidden');
  }
});

document.addEventListener('click', (e) => {
  if (!e.target.classList.contains('copy-code')) return;
  const code = e.target.parentElement.querySelector('code');
  navigator.clipboard.writeText(code.textContent).then(() => {
    e.target.textContent = 'copied';
    setTimeout(() => { e.target.textContent = 'copy'; }, 1400);
  });
});

// Drag-and-drop, with a visible target. dragleave fires constantly as the
// pointer crosses child elements, so track depth rather than trusting one event.
let dragDepth = 0;

function showDropzone() {
  if (document.querySelector('.dropzone')) return;
  const z = el('div', 'dropzone');
  z.innerHTML = '<div class="dropzone-card">' +
    '<div class="big">▤</div>' +
    '<div class="lab">Drop to attach</div>' +
    '<div class="sub">Code, CSV, Markdown, JSON, text or PDF</div></div>';
  document.body.appendChild(z);
}
function hideDropzone() {
  dragDepth = 0;
  const z = document.querySelector('.dropzone');
  if (z) z.remove();
}

document.addEventListener('dragenter', (e) => {
  if (!e.dataTransfer || !Array.from(e.dataTransfer.types).includes('Files')) return;
  e.preventDefault();
  dragDepth++;
  showDropzone();
});
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('dragleave', () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) hideDropzone();
});
document.addEventListener('drop', (e) => {
  e.preventDefault();
  hideDropzone();
  const files = e.dataTransfer ? Array.from(e.dataTransfer.files) : [];
  files.forEach(uploadFile);   // multiple at once, not just the first
});

boot().catch(e => {
  document.body.innerHTML =
    '<div class="empty-note" style="padding-top:20vh">Could not reach the server.<br>' +
    e + '</div>';
});
