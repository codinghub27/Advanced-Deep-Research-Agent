// Manual UI check: the sources display code in research_app/templates/chat.html
// (numbered [N] markers, the collapsible sources panel, the search-plan drawer), run in a DOM emulator.
// NOT part of the Python suite and NOT a project dependency:
//   npm install --no-save jsdom   (in any scratch folder)   then   node tests/ui/check_sources_ui.js [path-to-chat.html]
//   (set NODE_PATH to that folder's node_modules if jsdom is not next to this file)
// Only the block between the two markers is evaluated; the rest of the page (login, fetch, init) is not run.
const fs = require('fs');
const { JSDOM } = require('jsdom');

const html = fs.readFileSync(process.argv[2] || 'research_app/templates/chat.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/g).pop().replace(/^<script>|<\/script>$/g, '');
const start = script.indexOf('const SRC_META=');
const end = script.indexOf('/* ==========================================================================\n   PROGRESS STEPPER');
const block = script.slice(start, end);

const dom = new JSDOM('<!DOCTYPE html><body><div id="chatArea"></div><div id="ans"></div><div id="slot"></div><div id="slot2"></div><div id="routerLive"></div></body>', { url: 'http://localhost/' });
const { window } = dom;
global.window = window; global.document = window.document; global.NodeFilter = window.NodeFilter;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const chatArea = $('chatArea'); const scrollBottom = () => {};
// innerText and scrollIntoView are not implemented in jsdom
Object.defineProperty(window.HTMLElement.prototype, 'innerText', { get() { return this.textContent; } });
let scrolled = [];
window.HTMLElement.prototype.scrollIntoView = function () { scrolled.push(this.id); };

const api = new Function('$', 'esc', 'chatArea', 'scrollBottom', 'document', 'window', 'NodeFilter',
  block + '\nreturn {siteName,safeUrl,decorateAnswer,planHTML,renderSourcesFooter,renderRoutingLive,openPlanPanel,srcIcon,routeChip,highlightSource,knownType};')
  ($, esc, chatArea, scrollBottom, document, window, window.NodeFilter);

let fails = 0;
const ok = (cond, msg) => { if (!cond) { fails++; console.log('FAIL:', msg); } else console.log('ok  :', msg); };

// site names
ok(api.siteName('www.navtutorial.com') === 'navtutorial', 'siteName strips www and TLD');
ok(api.siteName('docs.fastapi.tiangolo.com') === 'tiangolo', 'siteName uses the registrable label');
ok(api.siteName('news.bbc.co.uk') === 'bbc', 'siteName handles co.uk');
ok(api.siteName('github.com') === 'github' && api.siteName('') === 'source', 'siteName basics');

// url safety
ok(api.safeUrl('javascript:alert(1)') === '#' && api.safeUrl('data:text/html,x') === '#', 'safeUrl rejects javascript:/data:');
ok(api.safeUrl('https://example.com/a?b=1') === 'https://example.com/a?b=1', 'safeUrl keeps https');
ok(api.knownType('official_docs') === 'official_docs' && api.knownType('rag_document') === 'web' && api.knownType(undefined) === 'web', 'unknown source types render as web');

const cites = [
  { index: 1, source_type: 'official_docs', title: 'Install <b>FastAPI</b>', url: 'https://fastapi.tiangolo.com/tutorial/', domain: 'fastapi.tiangolo.com', snippet: 'pip install fastapi' },
  { index: 2, source_type: 'github', title: 'repo', url: 'https://github.com/a/b', domain: 'github.com', snippet: 's' },
  { index: 3, source_type: 'reddit', title: 'thread', url: 'javascript:alert(1)', domain: 'www.reddit.com', snippet: 's' },
  { index: 4, source_type: 'web', title: 'A blog post', url: 'https://medium.com/@x/y', domain: 'medium.com', snippet: 's' },
];
const routing = [
  { task_id: 't1', sub_question: 'Install FastAPI <img src=x onerror=alert(1)>', source_intent: 'technical_howto', sources: ['official_docs', 'web'], origins: { official_docs: 'planner', web: 'policy' }, dropped: [{ source: 'reddit', reason: 'over_cap' }, { source: 'bogus', reason: 'unknown_source_type' }], stage: 'search',
    outcomes: [{ source: 'official_docs', status: 'ok', result_count: 3 }, { source: 'web', status: 'failed', result_count: 0, error_code: 'provider_error' }] },
  { task_id: 't1', sub_question: 'Install FastAPI', sources: ['web'], origins: {}, dropped: [], stage: 'gap_search', outcomes: [{ source: 'web', status: 'empty', result_count: 0 }] },
];

// panel (rendered first, as the page does; markers are then bound to it)
const slot = $('slot'); slot.id = 'src-r1';
api.renderSourcesFooter(slot, cites, routing);
const panel = slot.querySelector('.sources-panel');
ok(panel && /Sources \(4\)/.test(panel.querySelector('.sources-header').textContent), 'panel header shows "Sources (4)"');
ok(panel.querySelectorAll('.source-card').length === 4, 'one card per citation');
ok(panel.querySelector('.sources-list').hidden === true && panel.querySelector('.sources-header').getAttribute('aria-expanded') === 'false', 'several citations: collapsed by default');
const cards = [...panel.querySelectorAll('.source-card')];
ok(cards[0].id === 'src-r1-source-1' && cards[0].dataset.sourceType === 'official_docs', 'card ids are scoped to the answer');
ok(cards.map(c => c.querySelector('.source-badge').textContent).join('|') === 'Official docs|GitHub|Reddit|Web', 'badge text per source type');
ok(cards.map(c => c.querySelector('.source-badge').className.replace('source-badge ', '')).join('|') === 'official_docs|github|reddit|web', 'badge class per source type');
ok(cards[0].querySelector('.source-title').textContent === 'Install <b>FastAPI</b>' && !cards[0].querySelector('.source-title b'), 'title is text, not HTML');
ok(cards[0].querySelector('.source-domain').textContent === 'fastapi.tiangolo.com', 'domain shown');
ok(cards[0].target === '_blank' && /noopener/.test(cards[0].rel) && cards[0].getAttribute('href') === 'https://fastapi.tiangolo.com/tutorial/', 'card opens the URL in a new tab, noopener');
ok(cards[2].getAttribute('href') === '#', 'javascript: URL neutralised');
ok(!/<img|favicon|google\.com/i.test(panel.innerHTML), 'no img / favicon / google request in the panel');
ok(panel.querySelectorAll('.avatar').length === 4, 'letter avatars in place of favicons');

// header toggle
panel.querySelector('.sources-header').click();
ok(panel.querySelector('.sources-list').hidden === false && panel.querySelector('.sources-toggle').textContent === '▼', 'click expands');
panel.querySelector('.sources-header').click();
ok(panel.querySelector('.sources-list').hidden === true && panel.querySelector('.sources-toggle').textContent === '▶', 'click collapses again');

// [N] markers
const ans = $('ans');
ans.innerHTML = '<p>Install with pip [1][2]. Also [3] and a missing one [9].</p><p>Single marker style [4]</p><pre><code>arr[1] = x[2]</code></pre><p>Inline <code>y[1]</code> code and a <a href="#">link [1]</a>.</p>';
api.decorateAnswer(ans, cites, slot);
const refs = [...ans.querySelectorAll('sup.citation-ref')];
ok(refs.length === 4, `4 markers created (got ${refs.length})`);
ok(refs.map(r => r.textContent).join('') === '[1][2][3][4]' && refs.map(r => r.dataset.citation).join(',') === '1,2,3,4', 'markers keep their numbers');
ok(refs[0].dataset.sourceType === 'official_docs' && refs[1].dataset.sourceType === 'github' && refs[2].dataset.sourceType === 'reddit', 'markers carry the source type (colour)');
ok(refs[0].title === 'Install <b>FastAPI</b> — fastapi.tiangolo.com', 'hover tooltip = title + domain');
ok(ans.querySelector('pre').textContent === 'arr[1] = x[2]', 'code block untouched');
ok(ans.querySelector('p:nth-of-type(3) code').textContent === 'y[1]', 'inline code untouched');
ok(ans.querySelector('a').textContent === 'link [1]', 'link text untouched');
ok(/missing one \[9\]/.test(ans.textContent), 'marker with no citation stays as text');
ok(ans.dataset.plain.includes('[1][2]'), 'plain text with [N] kept for Copy Report');
ok(!/<img|favicon|google\.com/i.test(ans.innerHTML), 'no img / favicon / google request in the answer');

// clicking a marker: expand, scroll, highlight
scrolled = [];
refs[3].click();
ok(panel.querySelector('.sources-list').hidden === false, 'clicking [4] expands the collapsed panel');
ok(scrolled.includes('src-r1-source-4'), 'clicking [4] scrolls to its card');
ok(cards[3].classList.contains('highlighted') && !cards[0].classList.contains('highlighted'), 'only that card is highlighted');
refs[0].dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
ok(cards[0].classList.contains('highlighted'), 'Enter on a marker highlights too (keyboard)');
ok(refs[0].tabIndex === 0 && refs[0].getAttribute('role') === 'link', 'markers are focusable links');

// two answers in one chat do not interfere
const slot2 = $('slot2'); slot2.id = 'src-r2';
api.renderSourcesFooter(slot2, [cites[1], cites[3]].map((c, i) => ({ ...c, index: i + 1 })), []);
const panel2 = slot2.querySelector('.sources-panel');
ok(panel2.querySelector('.sources-list').hidden === true, 'second answer has its own collapsed panel');
ok(panel2.querySelector('.source-card').id === 'src-r2-source-1', 'second answer ids do not collide with the first');
api.highlightSource(slot2, 2);
ok(panel2.querySelector('.sources-list').hidden === false && panel2.querySelectorAll('.highlighted').length === 1 && panel2.querySelector('.highlighted').dataset.index === '2', 'highlight opens and marks the card in its own answer');
ok(!slot.querySelector('#src-r2-source-2') && slot.querySelectorAll('.highlighted').length === 2, 'the first answer is not affected');

// single citation: panel not collapsed
const slot3 = document.createElement('div'); slot3.id = 'src-r3'; document.body.appendChild(slot3);
api.renderSourcesFooter(slot3, [cites[0]], []);
ok(slot3.querySelector('.sources-list').hidden === false, 'single citation: panel is open by default');
ok(!slot3.querySelector('.src-footer'), 'no routing: no "Search plan" row');

// search plan button + drawer
const btn = slot.querySelector('.src-btn');
ok(btn && /Search plan/.test(btn.textContent), 'footer has a "Search plan" button');
ok(/Searched/.test(slot.textContent) && /Official docs/.test(slot.textContent), 'footer lists the sources the router searched');
btn.click();
const drawer = document.querySelector('.sp');
ok(drawer && /Search plan/.test(drawer.textContent) && drawer.querySelectorAll('.plan-q').length === 1, 'drawer shows the plan, grouped by sub-question');
ok(/over the per-question source limit/.test(drawer.textContent) && /not a supported source/.test(drawer.textContent), 'dropped suggestions shown with the reason');
ok(/Gap search/.test(drawer.textContent), 'gap-search stage labelled');
ok(drawer.querySelector('.rchip.bad') !== null && drawer.querySelector('.rchip.warn') !== null, 'failed / empty outcomes styled');
ok(!drawer.querySelector('img[onerror]'), 'sub-question text is escaped (no injected <img>)');
document.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape' }));
ok(!document.querySelector('.sp'), 'Escape closes the drawer');

// live router rows
api.renderRoutingLive(routing);
ok($('routerLive').querySelectorAll('.plan-q').length === 1 && !/Not used/.test($('routerLive').textContent), 'live router panel is compact');

// no citations
const slot4 = document.createElement('div'); slot4.id = 'src-r4'; document.body.appendChild(slot4);
api.renderSourcesFooter(slot4, [], routing);
ok(!slot4.querySelector('.sources-panel') && /Search plan/.test(slot4.querySelector('.src-btn').textContent), 'no citations: no sources panel, plan button only');
api.renderSourcesFooter(slot4, [], []);
ok(slot4.innerHTML === '', 'nothing to show: slot stays empty');
const bare = document.createElement('div'); bare.innerHTML = '<p>Answer [1] with no citations</p>';
api.decorateAnswer(bare, [], slot4);
ok(bare.innerHTML === '<p>Answer [1] with no citations</p>', 'no citations: answer text untouched');

console.log(fails ? `\n${fails} FAILED` : '\nALL UI CHECKS PASSED');
process.exit(fails ? 1 : 0);
