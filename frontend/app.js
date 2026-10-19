/* ============================================================
   GeoAgent 前端工作台
   面向使用：找数据 → 配任务 → 看执行 → 拿产物 → 翻记录
   ============================================================ */

const $ = (sel) => document.querySelector(sel);
const STORE_KEY = 'geoagent.session';

/* ---------------- 任务模板定义 ----------------
   objects 描述该模板「分析对象」下拉里可选的数据类别，与后端各工具的能力对齐：
   - index       ：仅派生指数（生态质量评估走指数现算，工具支持全部 8 个）
   - indexSeries ：派生指数 + 多期数据集（趋势/变化/制图可直接分析指数）
   - dataset     ：可直读为数据集的指数 + 多期数据集（时间序列/分区/报告只吃数据集）
   - all         ：派生指数 + 全部连续型数据集（统计概况，含 DEM/坡度等静态数据）
   - null        ：该模板不使用「分析对象」（情景预估、查看数据） */
const TASKS = [
  { key: 'index',      label: '生态质量评估',   subject: true,  objects: 'index',       range: false, year: true,  climate: false },
  { key: 'trend',      label: '变化趋势分析',   subject: true,  objects: 'indexSeries', range: true,  year: false, climate: false },
  { key: 'change',     label: '两期变化检测',   subject: true,  objects: 'indexSeries', range: true,  year: false, climate: false },
  { key: 'timeseries', label: '时间序列分析',   subject: true,  objects: 'dataset',     range: true,  year: false, climate: false },
  { key: 'zonal',      label: '按地类分区统计', subject: true,  objects: 'dataset',     range: false, year: true,  climate: false },
  { key: 'compose',    label: '专题图制作',     subject: true,  objects: 'indexSeries', range: false, year: true,  climate: false },
  { key: 'stats',      label: '统计概况',       subject: true,  objects: 'all',         range: false, year: true,  climate: false },
  { key: 'report',     label: '汇总分析报告',   subject: true,  objects: 'dataset',     range: false, year: false, climate: false },
  { key: 'scenario',   label: '未来情景预估',   subject: false, objects: null,          range: false, year: false, climate: true },
  { key: 'describe',   label: '查看可用数据',   subject: false, objects: null,          range: false, year: false, climate: false },
];
const TASK_LABELS = Object.fromEntries(TASKS.map((t) => [t.key, t.label]));

// 研究区名称：仅用于示例文案与建议任务的措辞，可按需替换为你自己的研究区
const STUDY_AREA = '研究区';

const STARTERS = [
  `评估 2020 年${STUDY_AREA}生态质量状况`,
  '分析 2000-2025 年 NDVI 的变化趋势与显著性',
  '对比 2000 年与 2020 年 NDVI 的变化情况',
  '统计 2020 年土地利用各类型的 NDVI',
  '未来不同情景下降水量将如何变化',
  `汇总一份${STUDY_AREA}植被覆盖分析报告`,
];

const ART_GROUPS = [
  { key: 'image',  label: '图件' },
  { key: 'data',   label: '数据与栅格' },
  { key: 'report', label: '报告' },
];

const state = {
  sessionId: null,
  busy: false,
  datasets: [],
  indices: [],
  years: [],
  tab: 'dataset',
  keyword: '',
  artifacts: [],
  seen: new Set(),
  history: [],
  turn: null,
  nodes: [],
};

/* ---------------- 启动 ---------------- */
async function boot() {
  bind();
  state.sessionId = localStorage.getItem(STORE_KEY) || null;
  $('#stream').innerHTML = introHtml();
  try {
    const meta = await (await fetch('/api/meta')).json();
    applyMeta(meta);
    buildBuilder();
  } catch (e) {
    $('#modeBadge').innerHTML = '<i class="pulse"></i><span>服务未就绪</span>';
    $('#assetList').innerHTML = '<p class="empty">未能连接后端服务，请确认已启动</p>';
    $('#assetHint').textContent = '';
    return;
  }
  renderAssets();
  await restoreSession();
  renderHistory();
  renderGallery();
}

function applyMeta(meta) {
  const badge = $('#modeBadge');
  const online = meta.mode === 'llm';
  badge.className = 'badge' + (online ? ' on' : '');
  badge.innerHTML = `<i class="pulse"></i><span>${online ? (meta.llm_model || '大模型') + ' 在线' : '离线规则模式'}</span>`;
  state.datasets = meta.datasets || [];
  state.indices = meta.derived_indices || [];
  state.years = collectYears();
}

function collectYears() {
  const set = new Set();
  state.datasets.forEach((d) => (d.years || []).forEach((y) => {
    if (y >= 1980 && y <= 2100) set.add(y);
  }));
  const list = Array.from(set).sort((a, b) => a - b);
  return list.length ? list : Array.from({ length: 26 }, (_, i) => 2000 + i);
}

function bind() {
  $('#composerForm').addEventListener('submit', (e) => {
    e.preventDefault();
    submit($('#input').value);
  });
  const input = $('#input');
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit(input.value);
    }
  });

  $('#resetBtn').addEventListener('click', resetSession);
  $('#zipBtn').addEventListener('click', () => {
    const url = $('#zipBtn').dataset.href;
    if (url) window.location.href = url;
  });

  // 数据资源
  $('#dataSearch').addEventListener('input', (e) => {
    state.keyword = e.target.value || '';
    renderAssets();
  });
  $('#dataTabs').addEventListener('click', (e) => {
    const tab = e.target.closest('.tab');
    if (!tab) return;
    state.tab = tab.dataset.tab;
    document.querySelectorAll('#dataTabs .tab').forEach((t) => t.classList.toggle('on', t === tab));
    renderAssets();
  });
  $('#assetList').addEventListener('click', (e) => {
    const btn = e.target.closest('.asset');
    if (btn) useAsset(btn.dataset.fill);
  });

  // 任务模板
  $('#builderToggle').addEventListener('click', () => {
    const body = $('#builderBody');
    const hidden = body.classList.toggle('folded');
    $('#builderToggle').textContent = hidden ? '展开' : '收起';
  });
  $('#fTask').addEventListener('change', () => { syncTaskFields(); updatePreview(); });
  $('#builderBody').addEventListener('change', updatePreview);
  $('#builderBody').addEventListener('input', updatePreview);
  $('#builderApply').addEventListener('click', () => {
    const text = buildTaskText();
    $('#input').value = text;
    $('#input').focus();
    $('#input').style.height = 'auto';
    $('#input').style.height = Math.min($('#input').scrollHeight, 120) + 'px';
  });

  // 引导模板
  $('#stream').addEventListener('click', (e) => {
    const btn = e.target.closest('.starter');
    if (btn) submit(btn.dataset.q);
  });

  // 任务记录
  $('#historyList').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-act]');
    if (!btn) return;
    const item = state.history[Number(btn.closest('.hist').dataset.i)];
    if (!item) return;
    if (btn.dataset.act === 'rerun') submit(item.query);
    else if (item.el) {
      item.el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      item.el.classList.remove('flash');
      void item.el.offsetWidth;
      item.el.classList.add('flash');
    }
  });

  // 灯箱
  $('#lightbox').addEventListener('click', () => { $('#lightbox').hidden = true; });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('#lightbox').hidden = true; });
}

/* ---------------- 数据资源 ---------------- */
function renderAssets() {
  const box = $('#assetList');
  const kw = state.keyword.trim().toLowerCase();

  if (state.tab === 'index') {
    const hits = state.indices.filter((i) => !kw
      || i.name.toLowerCase().includes(kw)
      || String(i.description || '').toLowerCase().includes(kw));
    box.innerHTML = hits.length
      ? assetGroup('派生指数', hits.map(indexNode))
      : '<p class="empty">没有匹配的指数</p>';
    $('#assetHint').textContent = `共 ${state.indices.length} 个可算指数${kw ? `，匹配 ${hits.length} 个` : ''} · 点击即引用到任务`;
    return;
  }

  const hits = state.datasets.filter((d) => !kw || matchDataset(d, kw));
  if (!hits.length) {
    box.innerHTML = '<p class="empty">没有匹配的数据集</p>';
  } else {
    const groups = new Map();
    hits.forEach((d) => {
      const tag = (d.tags && d.tags[0]) || '其他';
      if (!groups.has(tag)) groups.set(tag, []);
      groups.get(tag).push(datasetNode(d));
    });
    box.innerHTML = Array.from(groups, ([tag, items]) => assetGroup(tag, items)).join('');
  }
  $('#assetHint').textContent = `共 ${state.datasets.length} 个数据集${kw ? `，匹配 ${hits.length} 个` : ''} · 点击即引用到任务`;
}

function matchDataset(d, kw) {
  const hay = [d.title, d.variable, d.key, d.unit, d.description, (d.tags || []).join(' ')]
    .join(' ').toLowerCase();
  return hay.includes(kw);
}

function assetGroup(label, items) {
  return `<div class="asset-group">
    <div class="asset-group-head">${escape(label)}<em>${items.length}</em></div>
    <div class="asset-items">${items.join('')}</div>
  </div>`;
}

function datasetNode(d) {
  const span = d.years && d.years.length ? `${d.years[0]}–${d.years[d.years.length - 1]}` : '静态数据';
  return `<button class="asset" type="button" data-fill="${escape(shortTitle(d.title))}"
      title="${escape(d.description || d.title)}">
    <b>${escape(shortTitle(d.title))}</b>
    <span class="asset-desc">${escape(d.unit || d.variable || '')} · ${span}</span>
  </button>`;
}

function indexNode(i) {
  return `<button class="asset" type="button" data-fill="${escape(i.name)}"
      title="${escape(i.description || i.name)}">
    <b>${escape(i.name)}</b>
    <span class="asset-desc">${escape(truncate(i.description || '', 26))}</span>
  </button>`;
}

function useAsset(text) {
  if (!text) return;
  const input = $('#input');
  const cur = input.value.trim();
  input.value = cur ? `${cur} ${text}` : text;
  input.focus();
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 120) + 'px';
}

/* ---------------- 任务模板 ---------------- */
function buildBuilder() {
  $('#fTask').innerHTML = TASKS.map((t) => `<option value="${t.key}">${t.label}</option>`).join('');
  $('#taskHint').textContent = '选好条件，自动拼成一句任务描述';
  syncTaskFields();
  updatePreview();
}

function datasetByTitle(v) {
  return state.datasets.find((d) => shortTitle(d.title) === v);
}
function isSeries(d) {
  return d.kind === 'continuous' && (d.years || []).length >= 2;
}

/* 按模板能力给出可用的分析对象 */
function subjectPool(cfg) {
  if (!cfg.objects) return { indices: [], datasets: [] };
  // 指数若能被同名数据集直接读取（如 NDVI→ndvi），才可用于「只吃数据集」的模板
  const readable = (i) => state.datasets.some((d) => d.key === i.name.toLowerCase());
  const ok = (d) => {
    if (cfg.objects === 'index') return false;
    if (cfg.objects === 'all') return d.kind === 'continuous';
    return isSeries(d);
  };
  const indices = cfg.objects === 'dataset'
    ? state.indices.filter(readable)
    : state.indices.slice();
  return { indices, datasets: state.datasets.filter(ok) };
}

function fillSubjects() {
  const cfg = TASKS.find((t) => t.key === $('#fTask').value) || TASKS[0];
  const sel = $('#fSubject');
  const prev = sel.value;
  const pool = subjectPool(cfg);
  const idxOpts = pool.indices
    .map((i) => `<option value="${escape(i.name)}">${escape(i.name)}．${escape(truncate(i.description || '', 18))}</option>`)
    .join('');
  const dsOpts = pool.datasets
    .map((d) => `<option value="${escape(shortTitle(d.title))}">${escape(shortTitle(d.title))}</option>`)
    .join('');
  sel.innerHTML =
    (idxOpts ? `<optgroup label="派生指数">${idxOpts}</optgroup>` : '') +
    (dsOpts ? `<optgroup label="原始数据集">${dsOpts}</optgroup>` : '') ||
    '<option value="">（该模板无需对象）</option>';

  const names = [...pool.indices.map((i) => i.name), ...pool.datasets.map((d) => shortTitle(d.title))];
  if (names.includes(prev)) sel.value = prev;
  else if (names.includes('NDVI')) sel.value = 'NDVI';
  else if (names.length) sel.value = names[0];
}

/* 对象实际可用的年份（派生指数跟随其源数据 Reflectance/NDVI） */
function subjectYears(subject) {
  if (state.indices.some((i) => i.name === subject)) {
    const src = state.datasets.find((d) => d.key === 'reflectance') || state.datasets.find((d) => d.key === 'ndvi');
    const years = src && (src.years || []).length ? src.years : state.years;
    return years.slice().sort((a, b) => a - b);
  }
  const ds = datasetByTitle(subject);
  if (ds && (ds.years || []).length) return ds.years.slice().sort((a, b) => a - b);
  return state.years;
}

function fillYears() {
  const cfg = TASKS.find((t) => t.key === $('#fTask').value) || TASKS[0];
  const years = cfg.objects ? subjectYears($('#fSubject').value) : state.years;
  const list = years.length ? years : state.years;
  const opts = list.map((y) => `<option value="${y}">${y} 年</option>`).join('');
  const prevStart = Number($('#fStart').value);
  const prevEnd = Number($('#fEnd').value);
  const prevYear = Number($('#fYear').value);
  $('#fStart').innerHTML = opts;
  $('#fEnd').innerHTML = opts;
  $('#fYear').innerHTML = opts;
  const pick = (v, fb) => (list.includes(v) ? v : fb);
  $('#fStart').value = pick(prevStart, list[0]);
  $('#fEnd').value = pick(prevEnd, list[list.length - 1]);
  $('#fYear').value = pick(prevYear, list.includes(2020) ? 2020 : list[list.length - 1]);
}

function syncTaskFields() {
  const cfg = TASKS.find((t) => t.key === $('#fTask').value) || TASKS[0];
  [['#fieldSubject', cfg.subject], ['#fieldRange', cfg.range],
   ['#fieldYear', cfg.year], ['#fieldClimate', cfg.climate]]
    .forEach(([sel, show]) => { $(sel).hidden = !show; });
  if (cfg.subject) fillSubjects();
  fillYears();
}

function buildTaskText() {
  const task = $('#fTask').value;
  const subject = $('#fSubject').value || 'NDVI';
  const start = $('#fStart').value;
  const end = $('#fEnd').value;
  const year = $('#fYear').value;
  const climate = $('#fClimate').value === 'temp' ? '气温' : '降水量';
  const report = $('#fReport').checked ? '，并生成一份分析报告' : '';
  switch (task) {
    case 'index':      return `评估 ${year} 年${STUDY_AREA}${subject}的生态质量状况${report}`;
    case 'trend':      return `分析 ${start}-${end} 年${subject}的变化趋势与显著性${report}`;
    case 'change':     return `对比 ${start} 年与 ${end} 年${subject}的变化情况${report}`;
    case 'timeseries': return `分析 ${start}-${end} 年${subject}的区域平均变化${report}`;
    case 'zonal':      return `统计 ${year} 年土地利用各类型的${subject}${report}`;
    case 'compose':    return `绘制 ${year} 年${subject}的空间分布图${report}`;
    case 'stats':      return `统计 ${year} 年${subject}的分布与极值${report}`;
    case 'report':     return `汇总一份${STUDY_AREA}${subject}的分析报告`;
    case 'scenario':   return `未来不同情景下${climate}将如何变化${report}`;
    default:           return '目前有哪些数据可以使用';
  }
}

function updatePreview() {
  $('#builderPreview').textContent = '将生成：' + buildTaskText();
}

/* ---------------- 阶段指示 ---------------- */
const STAGES = ['understanding', 'recall', 'planning', 'acting', 'finalizing'];
function setRail(stage, finished = false) {
  const idx = STAGES.indexOf(stage);
  document.querySelectorAll('.rail-step').forEach((el, i) => {
    el.classList.toggle('live', !finished && i === idx);
    el.classList.toggle('done', finished ? true : (idx >= 0 && i < idx));
  });
}

/* ---------------- 提交任务 ---------------- */
async function submit(raw) {
  const query = (raw || '').trim();
  if (!query || state.busy) return;
  state.busy = true;
  $('#sendBtn').disabled = true;
  $('#input').value = '';
  $('#input').style.height = 'auto';

  const intro = $('#intro');
  if (intro) intro.remove();

  state.turn = createTurn(query);
  state.turn.hist = addHistory(query, state.turn.root);
  $('#stream').appendChild(state.turn.root);
  setRail('understanding');
  scrollToEnd();

  try {
    const resp = await fetch('/api/task', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, session_id: state.sessionId }),
    });
    if (!resp.ok || !resp.body) throw new Error('HTTP ' + resp.status);
    await consume(resp.body);
  } catch (err) {
    state.turn.cards.appendChild(el('div', 'err', '连接中断：' + err.message));
    finishTurn('fail');
  } finally {
    state.busy = false;
    $('#sendBtn').disabled = false;
    setRail(null, true);
  }
}

function finishTurn(status) {
  const t = state.turn;
  if (t && t.hist) {
    t.hist.status = status;
    renderHistory();
  }
}

async function resetSession() {
  try {
    const r = await (await fetch('/api/session/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId }),
    })).json();
    state.sessionId = r.session_id;
    localStorage.setItem(STORE_KEY, state.sessionId);
  } catch (e) { /* 忽略 */ }
  state.artifacts = [];
  state.seen.clear();
  state.history = [];
  state.nodes = [];
  state.turn = null;
  $('#stream').innerHTML = introHtml();
  setRail(null);
  renderHistory();
  renderGallery();
}

async function consume(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split('\n\n');
    buf = parts.pop();
    for (const chunk of parts) {
      const line = chunk.split('\n').find((l) => l.startsWith('data:'));
      if (!line) continue;
      try { handle(JSON.parse(line.slice(5).trim())); } catch (e) { /* 忽略坏帧 */ }
    }
  }
}

/* ---------------- 事件分发（只保留用户关心的信息） ---------------- */
function handle(ev) {
  const t = state.turn;
  if (!t) return;
  switch (ev.type) {
    case 'session':
      state.sessionId = ev.session_id;
      localStorage.setItem(STORE_KEY, ev.session_id);
      break;
    case 'status':
      t.status.textContent = plain(ev.text);
      setRail(ev.stage);
      break;
    case 'understanding':
      t.cards.appendChild(renderUnderstanding(ev.data));
      break;
    case 'recall':
      if (ev.memory || (ev.catalog && ev.catalog.length)) t.cards.appendChild(renderRecall(ev));
      break;
    case 'plan':
      t.cards.appendChild(renderPlan(ev.steps));
      break;
    case 'step_start':
      markNode(ev.tool, 'running');
      break;
    case 'progress': {
      const node = markNode(ev.tool, 'running');
      if (node) {
        node.querySelector('.progress-txt').textContent =
          ev.text + (ev.pct >= 0 ? `  ${Math.round(ev.pct)}%` : '');
      }
      break;
    }
    case 'tool_result': {
      const node = markNode(ev.data.name, ev.data.ok ? 'done' : 'fail');
      if (node) node.querySelector('.node-body').appendChild(renderResult(ev.data));
      (ev.data.artifacts || []).forEach(addArtifact);
      break;
    }
    case 'reflection':
      t.cards.appendChild(el('div', 'card',
        `<header><span class="dot"></span>自动调整</header><div class="md-body">${markdown(ev.text)}</div>`));
      break;
    case 'final':
      t.status.textContent = '交付完成';
      t.status.parentElement.classList.add('fin');
      t.root.appendChild(renderAnswer(ev.data));
      finishTurn('done');
      break;
    case 'error':
      t.cards.appendChild(el('div', 'err', escape(ev.text)));
      break;
    default:
      break;
  }
  scrollToEnd();
}

/* ---------------- 渲染片段 ---------------- */
function introHtml() {
  return `
  <div class="intro" id="intro">
    <p class="kicker">开箱即用的遥感分析</p>
    <h2>说出你要分析什么，剩下的交给它</h2>
    <p class="lede">左侧挑选数据、下方选用现成模板，或者直接用一句话描述需求。智能体会自己找数据、定方案、跑计算，最后把图件、表格和报告交到你手上。</p>
    <div class="starters">
      <p>常用分析，点一下就能开始：</p>
      <div class="starter-grid" id="starterGrid">
        ${STARTERS.map((q) => `<button class="starter" type="button" data-q="${escape(q)}">${escape(q)}</button>`).join('')}
      </div>
    </div>
  </div>`;
}

function createTurn(query) {
  const root = el('article', 'turn');
  root.innerHTML = `
    <div class="ask"><span class="who">任务</span><p>${escape(query)}</p></div>
    <div class="agent">
      <div class="statusline"><span class="spinner"></span><span class="txt">已提交，正在解析…</span></div>
      <div class="cards"></div>
    </div>`;
  return {
    root,
    cards: root.querySelector('.cards'),
    status: root.querySelector('.statusline .txt'),
    hist: null,
  };
}

function renderUnderstanding(u) {
  const chips = [];
  if (u.task_type) chips.push(chip('分析类型', TASK_LABELS[u.task_type] || u.task_type));
  if (u.indices && u.indices.length) chips.push(chip('指数', u.indices.join(' / ')));
  if (u.datasets && u.datasets.length) chips.push(chip('所用数据', u.datasets.map(datasetTitle).join(' / ')));
  if (u.start_year || u.end_year) chips.push(chip('时间', yearSpan(u.start_year, u.end_year)));
  if (u.deliverable) chips.push(chip('交付物', plain(u.deliverable), true));
  return el('div', 'card', `
    <header><span class="dot"></span>任务理解</header>
    <p class="goal">${inline(u.goal || '')}</p>
    <div class="kv">${chips.join('')}</div>`);
}

function renderRecall(ev) {
  const lines = [];
  if (ev.memory) lines.push('参考了本会话先前的分析经验');
  if (ev.catalog && ev.catalog.length) {
    lines.push('匹配到数据：' + ev.catalog
      .map((c) => escape(c.title ? shortTitle(c.title) : datasetTitle(c.key)))
      .join('、'));
  }
  return el('div', 'card', `
    <header><span class="dot"></span>匹配数据</header>
    <div class="goal dim">${lines.map((l) => `<p class="line">${l}</p>`).join('')}</div>`);
}

function renderPlan(steps) {
  state.nodes = [];
  const items = steps.map((s, i) => `
    <li class="node" data-tool="${escape(s.tool)}" data-i="${i}">
      <span class="dot"></span>
      <div class="node-body">
        <div class="node-head">
          <span class="idx">${String(i + 1).padStart(2, '0')}</span>
          <h4>${inline(s.title || s.tool)}</h4>
          <span class="state">待执行</span>
        </div>
        ${s.reason ? `<p class="reason">${inline(s.reason)}</p>` : ''}
        <div class="bar"><i></i></div>
        <p class="progress-txt"></p>
      </div>
    </li>`).join('');
  const card = el('div', 'card', `
    <header><span class="dot"></span>分析方案 · ${steps.length} 步</header>
    <ol class="timeline">${items}</ol>`);
  state.nodes = Array.from(card.querySelectorAll('.node'));
  return card;
}

function markNode(tool, status) {
  let node = state.nodes.find((n) => n.dataset.tool === tool && n.dataset.busy === '1');
  if (!node) node = state.nodes.find((n) => n.dataset.tool === tool && n.dataset.done !== '1');
  if (!node) return null;
  node.classList.remove('running', 'done', 'fail');
  node.classList.add(status);
  node.querySelector('.state').textContent = { running: '执行中', done: '已完成', fail: '未完成' }[status];
  if (status === 'running') {
    node.dataset.busy = '1';
  } else {
    node.dataset.busy = '0';
    node.dataset.done = '1';
  }
  return node;
}

function renderResult(rec) {
  const names = (rec.artifacts || []).map((a) => a.title).join('、');
  return el('div', 'result', `
    ${escape(rec.summary || '')}
    <span class="meta">用 ${rec.elapsed}s · 产出 ${(rec.artifacts || []).length} 项${names ? '：' + escape(truncate(names, 56)) : ''}</span>`);
}

function renderAnswer(payload) {
  const a = el('div', 'answer');
  const arts = payload.artifacts || [];
  a.innerHTML = `
    <h3>结果交付</h3>
    <div class="md-body">${markdown(payload.answer || '')}</div>
    <div class="footnote">
      <span>运行模式 · ${payload.mode === 'llm' ? '大模型' : '离线规则'}</span>
      <span>工具调用 · ${payload.count ? payload.count.tools : 0} 次</span>
      <span>产物 · ${arts.length} 项</span>
    </div>`;
  return a;
}

/* ---------------- 分析产物 ---------------- */
function addArtifact(art) {
  if (!art || !art.path || state.seen.has(art.path)) return;
  state.seen.add(art.path);
  state.artifacts.push(art);
  renderGallery();
}

function bucketOf(art) {
  if (art.type === 'image') return 'image';
  if (art.type === 'report') return 'report';
  const ext = extension(art);
  if (['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(ext)) return 'image';
  if (['md', 'html'].includes(ext)) return 'report';
  return 'data';
}

function renderGallery() {
  const box = $('#gallery');
  const zip = $('#zipBtn');

  if (!state.artifacts.length) {
    box.innerHTML = '<p class="empty">图件、数据表与报告会汇总在这里</p>';
  } else {
    const buckets = new Map(ART_GROUPS.map((g) => [g.key, []]));
    state.artifacts.forEach((a) => buckets.get(bucketOf(a)).push(a));
    box.innerHTML = ART_GROUPS
      .filter((g) => buckets.get(g.key).length)
      .map((g) => `<div class="art-group">
        <div class="art-group-head">${g.label}<em>${buckets.get(g.key).length}</em></div>
        ${buckets.get(g.key).map(artNode).join('')}
      </div>`).join('');
    box.querySelectorAll('img[data-full]').forEach((img) => {
      img.addEventListener('click', () => {
        $('#lightboxImg').src = img.dataset.full;
        $('#lightbox').hidden = false;
      });
    });
  }

  const ready = state.artifacts.length > 0 && !!state.sessionId;
  zip.disabled = !ready;
  zip.dataset.href = ready ? `/api/session/${state.sessionId}/artifacts.zip` : '';
}

function artNode(art) {
  const ext = extension(art);
  const isImg = bucketOf(art) === 'image';
  const label = { csv: '查看数据', md: '打开文档', html: '打开报告', tif: '下载栅格', geojson: '查看矢量' }[ext] || '打开文件';
  return `<div class="art">
    ${isImg ? `<img src="${art.url}" alt="${escape(art.title)}" loading="lazy" data-full="${art.url}">` : ''}
    <div class="cap">
      <b>${escape(art.title || art.path)}</b>
      <a href="${art.url}" target="_blank" rel="noopener">${label} ↗</a>
      ${ext ? `<span class="k">${ext.toUpperCase()}</span>` : ''}
    </div>
  </div>`;
}

/* ---------------- 任务记录 ---------------- */
function addHistory(query, turnEl) {
  const item = {
    id: 'h' + Date.now().toString(36),
    query,
    ts: timeLabel(),
    status: 'running',
    el: turnEl,
  };
  state.history.unshift(item);
  renderHistory();
  return item;
}

function renderHistory() {
  const box = $('#historyList');
  if (!state.history.length) {
    box.innerHTML = '<p class="empty">还没有执行过任务</p>';
    return;
  }
  box.innerHTML = state.history.map((h, i) => `
    <div class="hist" data-i="${i}">
      <div class="hist-top">
        <span class="hist-ts">${h.ts}</span>
        <span class="hist-state ${h.status}">${stateLabel(h.status)}</span>
      </div>
      <p class="hist-q">${escape(h.query)}</p>
      <div class="hist-act">
        <button class="link" type="button" data-act="view"${h.el ? '' : ' disabled'}>回看</button>
        <button class="link" type="button" data-act="rerun">重跑</button>
      </div>
    </div>`).join('');
}

function stateLabel(s) {
  return { running: '进行中', done: '已完成', fail: '失败', unknown: '历史记录' }[s] || '';
}

function timeLabel() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

async function restoreSession() {
  if (!state.sessionId) return;
  try {
    const resp = await fetch(`/api/session/${state.sessionId}`);
    if (!resp.ok) {
      state.sessionId = null;
      localStorage.removeItem(STORE_KEY);
      return;
    }
    const data = await resp.json();
    state.history = (data.messages || [])
      .filter((m) => m.role === 'user')
      .map((m, i) => ({ id: 'r' + i, query: m.content, ts: '历史', status: 'unknown', el: null }))
      .reverse();
    (data.artifacts || []).forEach(addArtifact);
  } catch (e) { /* 忽略 */ }
}

/* ---------------- 工具函数 ---------------- */
function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
}
function chip(k, v, sand) {
  return `<span class="${sand ? 'sand' : ''}">${escape(k)} <b>${escape(String(v))}</b></span>`;
}
function escape(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
/* 纯文本化：只能放纯文本的位置（状态栏、胶囊标签）剥掉 Markdown 记号，避免出现裸 ** # ` */
function plain(s) {
  return String(s == null ? '' : s)
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/`+/g, '')
    .replace(/\*\*/g, '')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/^\s{0,3}>\s?/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '')
    .trim();
}
function truncate(s, n) {
  s = String(s || '');
  return s.length > n ? s.slice(0, n) + '…' : s;
}
function shortTitle(t) {
  const s = String(t || '').split(/[（(]/)[0].trim();
  return s || String(t || '');
}
function datasetTitle(key) {
  const d = state.datasets.find((x) => x.key === key);
  return d ? shortTitle(d.title) : key;
}
function yearSpan(a, b) {
  if (a && b && a !== b) return `${a}–${b} 年`;
  if (a && b) return `${a} 年`;
  if (a) return `${a} 年起`;
  return `至 ${b} 年`;
}
function extension(art) {
  const m = String(art.path || art.url || '').toLowerCase().match(/\.([a-z0-9]+)$/);
  return m ? m[1] : '';
}
function scrollToEnd() {
  const s = $('#stream');
  s.scrollTop = s.scrollHeight;
}
function markdown(text) {
  const lines = String(text == null ? '' : text).replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let para = [];
  let list = null;      // { tag: 'ul'|'ol', items: [] }
  let table = null;     // { head: [], rows: [[]] }
  let quote = [];

  const flushPara = () => {
    if (!para.length) return;
    out.push(`<p>${inline(para.join('\n')).replace(/\n/g, '<br>')}</p>`);
    para = [];
  };
  const flushList = () => {
    if (!list) return;
    out.push(`<${list.tag}>` + list.items.map((t) => `<li>${inline(t)}</li>`).join('') + `</${list.tag}>`);
    list = null;
  };
  const flushTable = () => {
    if (!table) return;
    const head = table.head.map((c) => `<th>${inline(c)}</th>`).join('');
    const rows = table.rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`).join('');
    out.push(`<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`);
    table = null;
  };
  const flushQuote = () => {
    if (!quote.length) return;
    out.push(`<blockquote>${inline(quote.join('\n')).replace(/\n/g, '<br>')}</blockquote>`);
    quote = [];
  };
  const flushAll = () => { flushPara(); flushList(); flushTable(); flushQuote(); };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) { flushAll(); continue; }

    // 表格：形如 | a | b | 的表头 + |---|---| 分隔行
    if (line.startsWith('|') && line.endsWith('|') && i + 1 < lines.length
        && /^\|[\s:|-]+\|$/.test(lines[i + 1].trim())) {
      flushAll();
      table = { head: splitRow(line), rows: [] };
      i += 1;
      while (i + 1 < lines.length && lines[i + 1].trim().startsWith('|')) {
        i += 1;
        table.rows.push(splitRow(lines[i].trim()));
      }
      flushTable();
      continue;
    }

    // 标题：# ~ ######（正文层级整体下沉，避免与卡片标题抢层级）
    const head = line.match(/^(#{1,6})\s+(.+?)\s*#*$/);
    if (head) {
      flushAll();
      const lvl = head[1].length <= 2 ? 4 : (head[1].length === 3 ? 5 : 6);
      out.push(`<h${lvl}>${inline(head[2])}</h${lvl}>`);
      continue;
    }

    // 分隔线
    if (/^([-*_])\s*(\1\s*){2,}$/.test(line)) { flushAll(); out.push('<hr>'); continue; }

    // 引用
    if (line.startsWith('>')) {
      flushPara(); flushList(); flushTable();
      quote.push(line.replace(/^>\s?/, ''));
      continue;
    }

    // 无序列表
    const ul = line.match(/^[-*·•]\s+(.+)$/);
    if (ul) {
      flushPara(); flushTable(); flushQuote();
      if (!list || list.tag !== 'ul') { flushList(); list = { tag: 'ul', items: [] }; }
      list.items.push(ul[1]);
      continue;
    }

    // 有序列表
    const ol = line.match(/^\d+[.、)]\s*(.+)$/);
    if (ol) {
      flushPara(); flushTable(); flushQuote();
      if (!list || list.tag !== 'ol') { flushList(); list = { tag: 'ol', items: [] }; }
      list.items.push(ol[1]);
      continue;
    }

    flushList(); flushTable(); flushQuote();
    para.push(line);
  }
  flushAll();
  return out.join('');
}

function splitRow(line) {
  return line.replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
}

function inline(s) {
  return escape(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+|\/[^\s)]*)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>');
}

boot();
