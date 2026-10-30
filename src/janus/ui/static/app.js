/* JANUS UI. Built with DOM calls rather than HTML strings, so text from scenarios and models is never
   parsed as markup and the page stays within its content security policy (no inline styles or scripts). */
'use strict';

const view = document.getElementById('view');
const foot = document.getElementById('foot');
const statusBox = document.getElementById('status');
const navExplore = document.getElementById('nav-explore');
const COL_W = 262;
const ROW_H = 82;
const TOP = 44;
const GOLD = '#B08D57';

const state = { scenarios: [], tree: null, sens: null, option: null, selected: null, focal: null, formFor: null, job: null, status: null };

// ── Helpers ──────────────────────────────────────────────────────────────────

function h(tag, props, ...children) {
  const svg = tag.startsWith('svg:');
  const el = svg ? document.createElementNS('http://www.w3.org/2000/svg', tag.slice(4)) : document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue;
    if (key === 'style') for (const [prop, v] of Object.entries(value)) el.style.setProperty(prop, v);
    else if (key.startsWith('on')) el.addEventListener(key.slice(2).toLowerCase(), value);
    else el.setAttribute(key === 'className' ? 'class' : key, value === true ? '' : String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

async function api(path, options) {
  const response = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : response.statusText);
  return body;
}

const two = (n) => String(n).padStart(2, '0');
const prob = (x) => Number(x).toFixed(2);
const pct = (x) => `${Math.round(Number(x) * 100)}%`;
const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const fmt = (x) => (x == null ? '–' : Number(x).toFixed(1));
const upper = (s) => String(s).toUpperCase();

function secHead(num, title, note) {
  return h('div', { class: 'sec-head' },
    h('span', { class: 'num' }, num), h('span', { class: 'rule', 'aria-hidden': 'true' }),
    h('h2', {}, title), h('span', { class: 'fill', 'aria-hidden': 'true' }),
    note ? h('span', { class: 'note' }, note) : null);
}

function corners() {
  return ['tl', 'tr', 'bl', 'br'].map((c) => h('span', { class: `corner ${c}`, 'aria-hidden': 'true' }));
}

function setFoot(parts) {
  foot.replaceChildren(...parts.map((p) => h('span', {}, p)), h('span', { class: 'end' }, 'MS INTELLIGENCE'));
}

// ── Status ───────────────────────────────────────────────────────────────────

async function refreshStatus() {
  try {
    state.status = await api('/api/status');
  } catch {
    state.status = null;
  }
  const s = state.status;
  const chips = [];
  if (s) {
    const host = s.llmp ? new URL(s.llmp).host : 'NOT CONFIGURED';
    chips.push(h('span', { class: `chip ${s.llmp_ok ? 'on' : 'warn'}` },
      h('span', { class: `dot ${s.llmp_ok ? '' : 'off'}`, 'aria-hidden': 'true' }), `LLMP ${host} ${s.llmp_ok ? 'UP' : 'DOWN'}`));
    if (s.running) chips.push(h('span', { class: 'chip warn' }, `RUNNING ${upper(s.running.kind)} · ${s.running.spent}/${s.running.budget || '–'}`));
  }
  statusBox.replaceChildren(...chips);
}

// ── Routing ──────────────────────────────────────────────────────────────────

function setNav(active, name) {
  for (const link of document.querySelectorAll('.nav a')) {
    link.toggleAttribute('aria-current', link.dataset.view === active);
    if (link.dataset.view === active) link.setAttribute('aria-current', 'page');
  }
  navExplore.hidden = !name;
  if (name) navExplore.setAttribute('href', `#/explore/${encodeURIComponent(name)}`);
}

async function route() {
  const [, page, name] = location.hash.split('/');
  try {
    if (page === 'explore' && name) {
      setNav('explore', decodeURIComponent(name));
      await showExplorer(decodeURIComponent(name));
    } else {
      setNav('scenarios', state.tree && state.tree.name);
      await showScenarios();
    }
  } catch (error) {
    view.replaceChildren(h('p', { class: 'empty' }, `Could not load: ${error.message}`));
  }
  window.scrollTo(0, 0);
}

// ── Scenarios ────────────────────────────────────────────────────────────────

async function showScenarios() {
  state.scenarios = await api('/api/scenarios');
  renderScenarios();
}

function renderScenarios() {
  const list = state.scenarios.filter((s) => !s.error);
  const formFor = list.find((s) => s.name === state.formFor) || list[0];
  view.replaceChildren(
    h('div', { class: 'page-head' }, h('div', {},
      h('p', { class: 'kicker' }, `SCENARIOS · ${two(list.length)} FILES`),
      h('h1', { class: 'title' }, 'Scenarios'),
      h('p', { class: 'lede' }, 'Each scenario describes a world, the people or forces in it, and what each one wants. Explore one to see how its options hold up against chance and against each other.'))),
    h('section', { class: 'sec' },
      secHead('01', 'All scenarios'),
      h('div', { class: 'table-box' }, scenarioTable(formFor))),
    formFor ? h('section', { class: 'sec' }, secHead('02', `New exploration · ${formFor.name}`), exploreForm(formFor)) : null,
  );
  setFoot([`ENGINE v${state.status ? state.status.engine : '–'}`, 'MODELS RUN ON THIS MACHINE OR YOUR NETWORK ONLY']);
}

function scenarioTable(formFor) {
  const head = ['SCENARIO', 'AGAINST', 'HORIZON', 'PRIVACY', 'TREE'].map((t) => h('th', { scope: 'col' }, t));
  const rows = state.scenarios.map((s) => {
    if (s.error) return h('tr', { class: 'error-row' }, h('td', { colspan: '6' }, `${s.file}: ${s.error}`));
    const job = state.job && state.job.scenario === s.name && state.job.status === 'running' ? state.job : null;
    return h('tr', { class: formFor && s.name === formFor.name ? 'hl' : null },
      h('td', {}, h('div', { class: 'name' }, s.title), h('div', { class: 'sub' }, s.file)),
      h('td', {}, s.against === 'nature' ? 'Nature' : 'Other actors',
        h('div', { class: 'sub' }, s.against === 'nature' ? `${s.options.length} OPTIONS` : `${s.llm_actors.length} MODEL-PLAYED`)),
      h('td', { class: 'mono' }, `${s.clock.horizon} × ${upper(s.clock.unit)}`),
      h('td', {}, h('span', { class: `tag ${s.sensitive ? 'fail' : ''}` }, s.sensitive ? `${upper(s.privacy)} · SENSITIVE` : upper(s.privacy))),
      h('td', { class: 'mono' }, job ? jobCell(job) : s.tree
        ? [`${s.tree.nodes} NODES`, h('div', { class: 'sub' }, `COVERAGE ${pct(s.tree.coverage)} · ${(s.tree.updated || '').slice(0, 10)}`)]
        : h('span', { class: 'muted' }, 'NOT EXPLORED')),
      h('td', { class: 'actions' },
        s.tree ? h('a', { class: 'btn-primary', href: `#/explore/${encodeURIComponent(s.name)}` }, 'OPEN') : null,
        h('button', { class: 'btn', type: 'button', onClick: () => { state.formFor = s.name; renderScenarios(); } }, 'EXPLORE')));
  });
  return h('table', { class: 'list' }, h('thead', {}, h('tr', {}, head, h('th', {}, h('span', { class: 'hidden-label' }, 'Actions')))), h('tbody', {}, rows));
}

function jobCell(job) {
  const done = job.budget ? Math.min(30, Math.round((30 * job.spent) / job.budget)) : 0;
  return [h('span', { class: 'gold' }, job.kind === 'run' ? 'RUNNING' : 'EXPLORING'),
    h('div', { class: 'sub' }, job.budget ? `${job.spent} / ${job.budget} MODEL CALLS · ${job.nodes} NODES` : `${job.nodes} STEPS`),
    job.budget ? h('div', { class: 'progress', role: 'progressbar', 'aria-label': 'Exploration progress', 'aria-valuemin': '0',
      'aria-valuemax': String(job.budget), 'aria-valuenow': String(job.spent) },
    Array.from({ length: 30 }, (_, i) => h('i', { class: i < done ? 'on' : null }))) : null];
}

function exploreForm(s) {
  const field = (label, name, value, min) => h('label', {}, label,
    h('input', { type: 'number', name, value, min, inputmode: 'numeric' }));
  const message = h('p', { class: 'msg', role: 'status' });
  const running = state.job && state.job.status === 'running';
  const form = h('form', { class: 'console', onSubmit: (e) => { e.preventDefault(); startExplore(s, form, message); } },
    corners(),
    h('fieldset', {}, h('legend', {}, s.options.length ? 'OPTIONS' : 'ACTORS'),
      h('div', { class: 'opts' }, s.options.length
        ? s.options.map((label, i) => h('label', { class: 'opt' }, h('input', { type: 'checkbox', name: 'option', value: i, checked: true }),
          h('span', { class: 'mono muted' }, two(i + 1)), label))
        : s.llm_actors.map((a) => h('div', {}, h('span', { class: 'mono muted' }, 'LLM '), a, h('span', { class: 'p' }, ` ${s.models[`actor.${a}`]}`))))),
    h('div', { class: 'side' },
      h('div', { class: 'fields' },
        field('DEPTH', 'depth', 2, 1), field('BRANCHES', 'branches', 3, 1), field('ROLLOUTS', 'rollouts', 0, 0),
        field('BUDGET', 'budget', 40, 1)),
      h('p', { class: 'est' }, `GAME MASTER ${s.models.game_master} · ABOUT 30–60 S PER CALL · CACHED REPLIES FREE`, h('br'),
        s.privacy === 'local' ? 'LOCAL: MODELS RUN ON THIS MACHINE OR NETWORK · PROMPTS AND REPLIES NOT LOGGED' : 'CLOUD MODELS ALLOWED FOR THIS SCENARIO'),
      message,
      h('div', { class: 'buttons' },
        h('button', { class: 'btn-primary', type: 'submit', disabled: running }, 'RUN EXPLORATION'),
        h('button', { class: 'btn', type: 'button', disabled: running, onClick: () => startRun(s, form, message) }, 'RUN ONCE'))));
  return form;
}

function readForm(form) {
  const data = new FormData(form);
  const options = data.getAll('option').map(Number);
  return { options, depth: Number(data.get('depth')), branches: Number(data.get('branches')),
    rollouts: Number(data.get('rollouts')), budget: Number(data.get('budget')) };
}

async function startExplore(s, form, message) {
  const values = readForm(form);
  if (s.options.length && !values.options.length) { message.textContent = 'Choose at least one option.'; return; }
  await startJob(`/api/scenarios/${encodeURIComponent(s.name)}/explore`,
    { ...values, options: s.options.length ? values.options : null }, message);
}

async function startRun(s, form, message) {
  const values = readForm(form);
  await startJob(`/api/scenarios/${encodeURIComponent(s.name)}/run`, { option: values.options[0] || 0 }, message);
}

async function startJob(url, body, message) {
  message.textContent = '';
  try {
    state.job = await api(url, { method: 'POST', body: JSON.stringify(body) });
    renderScenarios();
    pollJob();
  } catch (error) {
    message.textContent = error.message;
  }
}

async function pollJob() {
  if (!state.job) return;
  state.job = await api(`/api/jobs/${state.job.id}`).catch(() => state.job);
  refreshStatus();
  const onScenarios = !location.hash.startsWith('#/explore');
  if (state.job.status === 'running') {
    if (onScenarios) renderScenarios();
    setTimeout(pollJob, 2000);
    return;
  }
  if (onScenarios) await showScenarios();
  else if (state.tree && state.tree.name === state.job.scenario) await showExplorer(state.job.scenario);
}

// ── Explorer ─────────────────────────────────────────────────────────────────

async function showExplorer(name) {
  const keepSelection = state.tree && state.tree.name === name;
  const query = state.focal ? `?focal=${encodeURIComponent(state.focal)}` : '';
  let tree;
  try {
    tree = await api(`/api/scenarios/${encodeURIComponent(name)}/tree${query}`);
  } catch (error) {
    view.replaceChildren(h('div', { class: 'empty' }, `${error.message}. `, h('a', { href: '#/' }, 'Explore it from the scenarios page.')));
    return;
  }
  state.tree = tree;
  if (!state.focal || !tree.actors.includes(state.focal)) state.focal = tree.actors[0] || null;
  if (!keepSelection || state.option == null || state.option >= tree.options.length) {
    state.option = bestOption(tree);
    state.selected = likeliestEnd(tree.options[state.option]);
  }
  state.sens = null;
  renderExplorer();
  const sensQuery = state.focal ? `?focal=${encodeURIComponent(state.focal)}` : '';
  api(`/api/scenarios/${encodeURIComponent(name)}/sensitivity${sensQuery}`).then((sens) => {
    if (state.tree && state.tree.name === name) { state.sens = sens; renderExplorer(); }
  }).catch(() => {});
}

function sensitivitySection() {
  const s = state.sens;
  if (!s) return null;
  const labels = s.options;
  const two2 = (x) => (x == null ? '–' : Number(x).toFixed(2));
  const flips = s.rows.filter((r) => r.changes_best);
  const closest = s.rows.reduce((c, r) => (r.margin != null && (!c || r.margin < c.margin) ? r : c), null);
  const summary = labels.length > 1
    ? `Best as explored: OPT ${two(labels.indexOf(s.baseline.best) + 1)}, ahead by ${two2(s.baseline.margin)}. `
      + `It changes under ${flips.length} of ${s.rows.length} changes${flips.length ? `: ${flips.map((r) => r.change).join('; ')}` : ''}.`
      + (closest ? ` Closest call: ${closest.change} (ahead by ${two2(closest.margin)}).` : '')
    : 'One course of action: the rows show how far its expected score moves.';
  const head = h('tr', {}, h('th', { scope: 'col' }, 'CHANGE'), labels.map((_, i) => h('th', { scope: 'col' }, `OPT ${two(i + 1)}`)),
    labels.length > 1 ? h('th', { scope: 'col' }, 'BEST') : null);
  const rows = [s.baseline, ...s.rows].map((row, index) => h('tr', { class: row.changes_best ? 'flip' : null },
    h('td', {}, index === 0 ? 'As explored' : cap(row.change)),
    labels.map((label) => h('td', { class: label === row.best && labels.length > 1 ? 'top' : null }, two2(row.expected[label]))),
    labels.length > 1 ? h('td', {}, row.changes_best ? h('span', { class: 'tag fail' }, `OPT ${two(labels.indexOf(row.best) + 1)}`)
      : h('span', { class: 'muted' }, index === 0 ? `OPT ${two(labels.indexOf(row.best) + 1)}` : 'same')) : null));
  return h('section', { class: 'sec' },
    secHead('03', 'What would change the answer', `SCORES FOR ${upper(s.focal)} · RE-WEIGHTED FROM THIS TREE, NO MODEL CALLS`),
    h('p', { class: 'lede sens-summary' }, summary),
    h('div', { class: 'table-box' }, h('table', { class: 'sens' }, h('thead', {}, head), h('tbody', {}, rows))),
    h('p', { class: 'note-line' }, 'Branches stay the ones explored; only their weights change. Changes that make unexplored outcomes likely lower the coverage.'));
}

const focalScore = (o) => (o.expected && state.focal in o.expected ? o.expected[state.focal] : null);

function bestOption(tree) {
  return tree.options.reduce((best, o, i) => ((focalScore(o) ?? -1) > (focalScore(tree.options[best]) ?? -1) ? i : best), 0);
}

function likeliestEnd(option) {
  const ends = option.nodes.filter((n) => n.kind === 'end');
  const pool = ends.length ? ends : option.nodes;
  const best = pool.reduce((b, n) => (!b || n.path_p > b.path_p ? n : b), null);
  return best ? best.id : null;
}

function leaves(option) {
  const parents = new Set(option.nodes.map((n) => n.parent));
  return option.nodes.filter((n) => !parents.has(n.id));
}

function renderExplorer() {
  const tree = state.tree;
  const option = tree.options[state.option];
  const best = bestOption(tree);
  const unit = upper(tree.clock.unit);
  const meanCoverage = tree.options.reduce((s, o) => s + o.coverage, 0) / Math.max(1, tree.options.length);
  view.replaceChildren(
    h('div', { class: 'page-head' },
      h('div', {},
        h('p', { class: 'kicker' }, `SCN ${tree.name} · ${tree.clock.horizon} × ${unit}`),
        h('h1', { class: 'title' }, tree.title)),
      h('dl', { class: 'kpis' },
        h('dt', {}, 'OPTIONS'), h('dt', {}, 'NODES'), h('dt', {}, 'COVERAGE'),
        h('dd', {}, two(tree.options.length)), h('dd', {}, tree.tree.nodes), h('dd', {}, pct(meanCoverage)))),
    h('section', { class: 'sec' },
      secHead('01', 'Options compared', 'SCORE / 10 · PROBABILITY-WEIGHTED'),
      tree.actors.length > 1 ? h('div', { class: 'segmented', role: 'group', 'aria-label': 'Whose objectives' },
        h('span', {}, 'SCORES FOR'),
        tree.actors.map((a) => h('button', { type: 'button', 'aria-pressed': String(a === state.focal),
          onClick: () => { state.focal = a; showExplorer(tree.name); } }, upper(a)))) : null,
      h('div', { class: 'cards' }, tree.options.map((o, i) => optionCard(o, i, i === best))),
      h('p', { class: 'note-line' }, 'Strip: every ending on a 0–10 scale · tick height = probability')),
    h('section', { class: 'sec' },
      secHead('02', `Ramifications · OPT ${two(state.option + 1)}`, 'LINE WEIGHT = PROBABILITY · DASHED = CARRIED TO THE END'),
      h('div', { class: 'explore-row' },
        h('figure', { class: 'tree-figure' }, h('div', { class: 'tree-scroll' }, treeCanvas(option)),
          h('figcaption', { class: 'caption' }, "Probabilities come from the game master and the scenario's base rates: relative weights, not forecasts.")),
        detailPanel(option))),
    sensitivitySection(),
  );
  setFoot([`TREE ${tree.tree.file}`,
    tree.tree.current ? `ENGINE v${tree.tree.engine}` : `BUILT BY ENGINE v${tree.tree.engine} · NEW EXPLORATIONS START A NEW TREE`,
    `UNFINISHED BRANCHES ${option.unfinished}`]);
}

function optionCard(o, i, isBest) {
  const ends = leaves(o);
  const top = Math.max(0.0001, ...ends.map((n) => n.path_p));
  const score = focalScore(o);
  return h('button', { class: 'card', type: 'button', 'aria-pressed': String(i === state.option),
    onClick: () => { state.option = i; state.selected = likeliestEnd(o); renderExplorer(); } },
  h('span', { class: 'card-top' }, h('span', {}, `OPT ${two(i + 1)}`), h('span', { class: 'tag-text gold' }, isBest ? 'HIGHEST E[SCORE]' : '')),
  h('span', { class: 'card-label' }, o.label),
  h('span', { class: 'figure-row' }, h('span', { class: `figure ${isBest ? 'best' : ''}` }, fmt(score)), h('span', { class: 'figure-unit' }, 'E[SCORE]')),
  h('span', { class: 'spectrum', 'aria-hidden': 'true' },
    ends.filter((n) => state.focal in n.score).map((n) => h('span', { class: 'tick', style: {
      left: `${Math.min(99, n.score[state.focal] * 10)}%`, height: `${Math.max(4, Math.round(28 * Math.sqrt(n.path_p / top)))}px` } }))),
  h('span', { class: 'stats' },
    h('span', {}, 'WORST 10%', h('b', {}, fmt(o.worst10 && o.worst10[state.focal]))),
    h('span', {}, 'ENDS EARLY', h('b', {}, pct(o.p_end))),
    h('span', {}, 'COVERAGE', h('b', {}, pct(o.coverage)))));
}

function treeCanvas(option) {
  const byId = Object.fromEntries(option.nodes.map((n) => [n.id, n]));
  const kids = {};
  for (const n of option.nodes) (kids[n.parent] = kids[n.parent] || []).push(n);
  for (const list of Object.values(kids)) list.sort((a, b) => b.p - a.p);
  const maxStep = Math.max(1, ...option.nodes.filter((n) => n.kind !== 'end').map((n) => n.step));
  const colOf = (n) => (n.kind === 'end' ? maxStep + 1 : n.step);
  // Leaves get rows depth-first (likeliest first); parents sit midway between their first and last child.
  const pos = {};
  let row = 0;
  const place = (key) => {
    const children = kids[key] || [];
    if (!children.length) { const y = TOP + row * ROW_H; row += 1; return y; }
    const ys = children.map((c) => { const y = place(c.id); pos[c.id] = { x: 24 + colOf(c) * COL_W, y }; return y; });
    return (ys[0] + ys[ys.length - 1]) / 2;
  };
  const rootY = place('root');
  const width = 24 + (maxStep + 2) * COL_W;
  const height = TOP + Math.max(row, 1) * ROW_H + 8;
  const lit = new Set();
  for (let id = state.selected; id && byId[id]; id = byId[id].parent) lit.add(id);

  const edges = h('svg:svg', { class: 'edges', width, height, 'aria-hidden': 'true' },
    option.nodes.map((n) => {
      const child = pos[n.id];
      const fromRoot = n.parent === 'root';
      const parent = fromRoot ? { x: 24, y: rootY } : pos[n.parent];
      const x1 = fromRoot ? 24 + 190 : parent.x + 210;
      const y1 = parent.y + 32;
      const y2 = child.y + 32;
      const mx = (x1 + child.x) / 2;
      return h('svg:path', { d: `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${child.x} ${y2}`, fill: 'none',
        stroke: lit.has(n.id) ? GOLD : 'rgba(245,241,232,0.22)', 'stroke-width': (1 + 5 * Math.sqrt(n.path_p)).toFixed(1),
        'stroke-dasharray': n.kind === 'end' ? '4 4' : null });
    }));

  const columns = [h('span', { class: 'col-label', style: { left: '24px' } }, 'T+00  DECISION')];
  for (let s = 1; s <= maxStep; s += 1) {
    columns.push(h('span', { class: 'col-label', style: { left: `${24 + s * COL_W}px` } }, `T+${two(s)}  ${upper(state.tree.clock.unit)} ${two(s)}`));
  }
  columns.push(h('span', { class: 'col-label', style: { left: `${24 + (maxStep + 1) * COL_W}px` } }, 'END'));

  const nodes = option.nodes.map((n) => {
    const score = n.score[state.focal];
    const cut = n.kind === 'end' && !n.terminal;  // carried forward until the budget ran out, not to the end
    const what = n.kind !== 'end' ? cap(n.what)
      : cut ? 'Stopped early: budget'
        : (n.via && n.via.events.length ? n.via.events.join(', ') : 'No surprises after this');
    return h('button', { type: 'button', class: `node ${lit.has(n.id) ? 'lit' : ''}`, 'aria-pressed': String(n.id === state.selected),
      'aria-label': `${n.label}, probability ${pct(n.path_p)}, ${what}, score ${fmt(score)}`,
      style: { left: `${pos[n.id].x}px`, top: `${pos[n.id].y}px` },
      onClick: () => { state.selected = n.id; renderExplorer(); } },
    h('span', { class: 'k' }, h('span', {}, `${n.kind !== 'end' ? `T+${two(n.step)}` : cut ? `STOP T+${two(n.step)}` : 'END'} · p ${prob(n.path_p)}`), h('span', { class: 'score' }, fmt(score))),
    h('span', { class: 'w' }, what),
    h('span', { class: 'meter', 'aria-hidden': 'true' }, h('i', { style: { width: `${Math.round((score || 0) * 10)}%` } })));
  });

  return h('div', { class: 'tree', style: { width: `${width}px`, height: `${height}px` } }, edges, columns,
    h('div', { class: 'root-node', style: { top: `${rootY}px` } },
      h('span', { class: 'k' }, `DECISION · OPT ${two(state.option + 1)}`), h('span', { class: 'v' }, option.label)),
    nodes);
}

function detailPanel(option) {
  const node = option.nodes.find((n) => n.id === state.selected);
  if (!node) return h('aside', { class: 'panel' }, corners(), h('p', { class: 'muted' }, 'Select a box in the tree.'));
  const unit = state.tree.clock.unit;
  const objectives = state.tree.objectives[state.focal] || [];
  const focalScores = node.scores[state.focal] || {};
  const others = Object.entries(node.score).filter(([a]) => a !== state.focal);
  const byActor = {};
  for (const r of node.orders) (byActor[r.actor] = byActor[r.actor] || []).push(r);
  const cut = node.kind === 'end' && !node.terminal;
  const surprises = node.kind === 'end' && node.via.events.length ? `, with ${node.via.events.join(', ')}` : ', with no surprises';
  return h('aside', { class: 'panel', 'aria-labelledby': 'detail-title' }, corners(),
    h('div', { class: 'panel-top' }, h('span', { class: 'gold' }, node.kind !== 'end' ? `T+${two(node.step)} · ${upper(node.label)}`
      : `T+${two(node.step)} · ${cut ? 'STOPPED' : 'END'}`),
    h('span', {}, `NODE ${node.id.slice(0, 8).toUpperCase()}`)),
    h('h3', { id: 'detail-title' }, node.kind !== 'end' ? cap(node.what) : cut ? `Stopped at ${node.label}` : `End: ${node.label}`),
    h('div', { class: 'readout' },
      h('span', { class: 'big' }, fmt(node.score[state.focal])),
      h('span', {}, 'P(PATH) ', h('b', {}, prob(node.path_p))),
      h('span', {}, 'P(BRANCH) ', h('b', {}, prob(node.p)))),
    h('p', { class: 'meta' }, node.kind !== 'end'
      ? `${pct(node.p)} of its branch; ${pct(node.path_p)} of all courses of events under this option.`
      : cut ? `Carried ${node.via.steps} ${unit}s${surprises}, then the budget ran out: scored as it stands, not at the end.`
        : `Carried ${node.via.steps} ${unit}s to the end${surprises}.`,
    others.length ? ` Others: ${others.map(([a, s]) => `${a} ${fmt(s)}`).join(', ')}.` : ''),
    h('div', { class: 'objectives' }, objectives.map(({ text, weight }) => {
      const s = focalScores[text] ?? 0;
      const label = weight === 1 ? text : `${text} (×${weight})`;
      return h('div', { class: 'objective' }, h('span', { class: 'row' }, h('span', {}, label), h('b', {}, fmt(s))),
        h('span', { class: 'bar4', 'aria-hidden': 'true' }, h('i', { style: { width: `${Math.round(s * 10)}%` } })));
    })),
    Object.entries(byActor).map(([actor, rolls]) => [
      h('h4', {}, `Orders · ${actor}`),
      h('ul', { class: 'items' }, rolls.map((r) => h('li', {}, h('span', { class: `tag ${r.worked ? '' : 'fail'}` }, r.worked ? 'WORKED' : 'FAILED'),
        h('span', {}, r.text, ' ', h('span', { class: 'p' }, `p ${prob(r.p)}`)))))]),
    node.events.length ? [h('h4', {}, 'Chance events'), h('ul', { class: 'items' },
      node.events.map((e) => h('li', {}, h('span', { class: 'tag' }, e.source === 'open' ? 'RAISED' : 'BASE RATE'), `${e.name} `, h('span', { class: 'p' }, `p ${prob(e.p)}`))))] : null,
    node.discoveries.length ? [h('h4', {}, 'Discoveries'), h('ul', { class: 'items' },
      node.discoveries.map((d) => h('li', {}, `${d.by} learned ${d.of}'s ${d.what}`)))] : null,
    node.kind !== 'end' && node.narrative !== 'Nothing notable happened.' ? h('p', { class: 'narrative' }, node.narrative) : null,
    Object.keys(node.observations).length ? [h('h4', {}, 'Private notes'), h('ul', { class: 'items' },
      Object.entries(node.observations).map(([a, t]) => h('li', {}, h('span', { class: 'tag' }, upper(a)), t)))] : null,
    h('h4', {}, 'State'),
    h('table', { class: 'state' }, h('tbody', {}, Object.entries(node.state).map(([k, v]) => h('tr', {}, h('th', { scope: 'row' }, k), h('td', {}, String(v)))))));
}

// ── Start ────────────────────────────────────────────────────────────────────

window.addEventListener('hashchange', route);
refreshStatus().then(async () => {
  const jobs = await api('/api/jobs').catch(() => []);
  state.job = jobs.find((j) => j.status === 'running') || null;
  if (state.job) pollJob();
  await route();
});
setInterval(refreshStatus, 10000);
