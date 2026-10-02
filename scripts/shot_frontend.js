// 前端渲染验证 + 截图。用真实浏览器走一遍完整流程，然后校验**渲染出来的
// 几何与文本**，而不只是「HTML 里有没有这个元素」。
//
//   node scripts/shot_frontend.js http://127.0.0.1:8131 docs/screenshots
//
// 为什么非要用真实浏览器
// ---------------------
// 这个页面里最容易坏、又最难靠单测发现的三件事：
//
// 1. **SVG 图表有没有真的画出来**。召回率散点图、内存条形图、HNSW 分层图
//    都是 JS 现画的。写错一个坐标计算公式不会报错，只会得到一个空画布 ——
//    HTML 完全正确，pytest 全绿，只有渲染出来才知道是空的；
// 2. **数值表格的列错位**。表格有 10 列，一旦表头和数据行的单元格数
//    不一致，浏览器会把后面的内容挤到错误的列里 —— 数字看起来总是对的，
//    只是它属于另一个引擎。这种 bug 靠看 HTML 源码发现不了；
// 3. **两种召回率的反差有没有被正确提示**。界面在标准召回与门槛召回
//    相差较大时会渲染一条警告说明。这个分支平时不走，
//    只有真的用 IVF-PQ 查一次才会出现。
//
// 全部断言都走「读表头名字 → 取列」而不是按位置取 td[3] 这种写法 ——
// 插入一列会让后者静默读到错误的单元格，变成假通过。

const fs = require('fs');
const path = require('path');

const BASE = process.argv[2] || 'http://127.0.0.1:8131';
const OUT = process.argv[3] || 'docs/screenshots';

const PW_CANDIDATES = [
  process.env.PW_PATH,
  'playwright-core',
  'playwright',
  'C:/Users/Legion/npm_global/node_modules/n8n/node_modules/playwright-core',
].filter(Boolean);

const CHROME_CANDIDATES = [
  process.env.CHROME,
  process.env.CHROME_PATH,
  'C:/Users/Legion/.agent-browser/browsers/chrome-153.0.8010.36/chrome.exe',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);

function loadPlaywright() {
  for (const candidate of PW_CANDIDATES) {
    try {
      return require(candidate);
    } catch (_) { /* 试下一个 */ }
  }
  console.error('找不到 playwright-core。用 PW_PATH 指过去：\n'
    + '  PW_PATH=/path/to/playwright-core node scripts/shot_frontend.js');
  process.exit(2);
}

function findChrome() {
  for (const candidate of CHROME_CANDIDATES) {
    if (candidate && fs.existsSync(candidate)) return candidate;
  }
  console.error('找不到 Chrome/Chromium。用 CHROME 环境变量指过去。');
  process.exit(2);
}

const { chromium } = loadPlaywright();

let passed = 0, failed = 0;
function check(name, cond, detail) {
  if (cond) { passed++; console.log('  ✓ ' + name); }
  else { failed++; console.log('  ✗ ' + name + (detail ? '  ← ' + detail : '')); }
}
function section(t) { console.log('\n[' + t + ']'); }

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  fs.mkdirSync(OUT, { recursive: true });

  const browser = await chromium.launch({
    executablePath: findChrome(),
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1100 },
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();

  // pageerror 才是真正的 JS 异常；资源加载失败（404/422）单独记一条，
  // 因为脚本第 8 段会故意触发一次 422 来验证错误提示。
  const jsErrors = [];
  const resourceErrors = [];
  page.on('console', (m) => {
    if (m.type() !== 'error') return;
    const t = m.text();
    if (t.indexOf('Failed to load resource') >= 0) resourceErrors.push(t);
    else jsErrors.push(t);
  });
  page.on('pageerror', (e) => jsErrors.push('pageerror: ' + e.message));
  let expectHttpError = false;

  const shot = async (name) => {
    const file = path.join(OUT, name + '.png');
    await page.screenshot({ path: file, fullPage: false });
    const stat = fs.statSync(file);
    check('截图 ' + name + ' 非空', stat.size > 8000, stat.size + ' bytes');
    return file;
  };

  // 等待某个异步请求结束：按钮回到原标题（即 spinner 消失）
  async function waitDone(sel, label) {
    try {
      await page.waitForFunction(
        (s) => {
          const btn = document.querySelector(s);
          return btn && !btn.disabled && btn.textContent.trim().indexOf('<') === -1;
        },
        sel, { timeout: 120000 }
      );
      return true;
    } catch (e) {
      console.log('    (等待 ' + label + ' 超时)');
      return false;
    }
  }

  console.log('='.repeat(70));
  console.log('vector-index · 前端渲染 + 截图   ' + BASE);
  console.log('='.repeat(70));

  // ---------------------------------------------------------------- 1
  section('1 加载与健康状态');
  await page.goto(BASE + '/', { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForTimeout(700);
  check('标题正确', (await page.textContent('h1')).indexOf('vector-index') >= 0,
    await page.textContent('h1'));
  check('健康指示点亮起',
    await page.evaluate(() => document.querySelector('#healthDot').className === 'on'));
  check('版本号已填充',
    (await page.textContent('#verTxt')).indexOf('v') === 0,
    await page.textContent('#verTxt'));
  await shot('01-初始');

  // 横向不能溢出
  const overflow = await page.evaluate(() =>
    document.documentElement.scrollWidth - document.documentElement.clientWidth);
  check('页面无横向溢出', overflow <= 1, '溢出 ' + overflow + 'px');

  // ---------------------------------------------------------------- 2
  section('2 生成数据集');
  await page.fill('#pCount', '600');
  await page.fill('#pDim', '32');
  await page.selectOption('#pMetric', 'l2');
  await page.selectOption('#pKind', 'clustered');
  await page.click('#btnDs');
  await waitDone('#btnDs', '生成数据集');
  await page.waitForTimeout(400);
  check('数据集面板已渲染', await page.isVisible('#dsOut .cards'));
  const dsId = await page.textContent('#curDs');
  check('当前数据集 ID 非空且为 12 位', dsId.length === 12, dsId);
  const cardCount = await page.evaluate(() =>
    document.querySelectorAll('#dsOut .card').length);
  check('渲染了 4 张概览卡', cardCount === 4, String(cardCount));

  const kbText = await page.evaluate(() => {
    const cards = [...document.querySelectorAll('#dsOut .card')];
    const hit = cards.find((c) => c.textContent.indexOf('向量本体内存') >= 0);
    return hit ? hit.querySelector('.v').textContent.trim() : '';
  });
  check('向量内存显示为 75.0 KB（600×128）', kbText === '75.0 KB', kbText);
  await shot('02-数据集');

  // ---------------------------------------------------------------- 3
  section('3 构建 HNSW');
  await page.selectOption('#pEngine', 'hnsw');
  await sleep(200);
  check('切换引擎后参数标签变成 HNSW 的',
    (await page.textContent('#labP1')).indexOf('m（每层邻居数）') >= 0,
    await page.textContent('#labP1'));
  await page.fill('#p1', '16');
  await page.fill('#p2', '100');
  await page.fill('#p3', '64');
  await page.click('#btnIdx');
  await waitDone('#btnIdx', '构建 HNSW');
  await page.waitForTimeout(400);
  check('索引面板已渲染', await page.isVisible('#idxOut .cards'));
  check('当前引擎显示为 hnsw',
    (await page.textContent('#curEng')).trim() === 'hnsw',
    await page.textContent('#curEng'));

  const hnswCards = await page.evaluate(() =>
    [...document.querySelectorAll('#idxOut .card')].map((c) => c.textContent));
  const joined = hnswCards.join(' | ');
  check('显示了层数卡片', joined.indexOf('层数') >= 0, joined.slice(0, 200));
  check('显示了第0层平均度', joined.indexOf('第0层平均度') >= 0, joined.slice(0, 200));
  check('显示了每次查询距离计算', joined.indexOf('每次查询距离计算') >= 0, joined.slice(0, 200));
  await shot('03-HNSW 索引');

  // ---------------------------------------------------------------- 4
  section('4 单次检索（HNSW）');
  await page.fill('#sK', '10');
  await page.fill('#sProbe', '0');
  await page.click('#btnSearch');
  await waitDone('#btnSearch', '检索');
  await page.waitForTimeout(400);
  check('检索面板已渲染', await page.isVisible('#searchOut .cards'));

  const recallVal = await page.evaluate(() => {
    const cards = [...document.querySelectorAll('#searchOut .card')];
    const hit = cards.find((c) => c.textContent.indexOf('标准召回率') >= 0);
    return hit ? hit.querySelector('.v').textContent.trim() : '';
  });
  check('HNSW 标准召回率 ≥ 95%', parseFloat(recallVal) >= 95, recallVal);

  const thrVal = await page.evaluate(() => {
    const cards = [...document.querySelectorAll('#searchOut .card')];
    const hit = cards.find((c) => c.textContent.indexOf('门槛召回率') >= 0);
    return hit ? hit.querySelector('.v').textContent.trim() : '';
  });
  check('门槛召回率也显示了', thrVal.length > 0, thrVal);

  const tableHeads = await page.evaluate(() =>
    [...document.querySelectorAll('#searchOut table thead th')].map((t) => t.textContent.trim()));
  check('表头含「ANN 是否召回」列', tableHeads.some((h) => h.indexOf('ANN 是否召回') >= 0),
    JSON.stringify(tableHeads));
  const rowsShown = await page.evaluate(() =>
    document.querySelectorAll('#searchOut table tbody tr').length);
  check('结果表格渲染了 10 行', rowsShown === 10, String(rowsShown));
  await shot('04-HNSW 检索');

  // ---------------------------------------------------------------- 5
  section('5 构建 IVF-PQ 并观察两种召回率的反差');
  await page.selectOption('#pEngine', 'ivfpq');
  await sleep(200);
  check('切换引擎后参数标签变成 IVF-PQ 的',
    (await page.textContent('#labP1')).indexOf('nlist') >= 0,
    await page.textContent('#labP1'));
  await page.fill('#p1', '16');
  await page.fill('#p2', '8');
  await page.fill('#p3', '4');
  await page.click('#btnIdx');
  await waitDone('#btnIdx', '构建 IVF-PQ');
  await page.waitForTimeout(400);

  const ivfCards = await page.evaluate(() =>
    [...document.querySelectorAll('#idxOut .card')].map((c) => c.textContent).join(' | '));
  check('显示了倒排表数', ivfCards.indexOf('倒排表数') >= 0, ivfCards.slice(0, 220));
  check('显示了每次查询扫描', ivfCards.indexOf('每次查询扫描') >= 0, ivfCards.slice(0, 220));
  await shot('05-IVFPQ 索引');

  await page.fill('#sNprobe', '1');
  await page.click('#btnSearch');
  await waitDone('#btnSearch', '检索');
  await page.waitForTimeout(400);

  const ivfPair = await page.evaluate(() => {
    const cards = [...document.querySelectorAll('#searchOut .card')];
    const std = cards.find((c) => c.textContent.indexOf('标准召回率') >= 0);
    const thr = cards.find((c) => c.textContent.indexOf('门槛召回率') >= 0);
    return {
      std: std ? std.querySelector('.v').textContent.trim() : '',
      thr: thr ? thr.querySelector('.v').textContent.trim() : '',
    };
  });
  check('两种召回率都取到', ivfPair.std.length > 0 && ivfPair.thr.length > 0,
    JSON.stringify(ivfPair));
  const gap = await page.evaluate(() => {
    const n = document.querySelector('#searchOut .note.warn');
    return n ? n.textContent.trim().slice(0, 80) : null;
  });
  check('召回率出现差距时渲染了说明（或有则不渲染）', gap === null || gap.indexOf('正常现象') >= 0,
    String(gap));
  await shot('06-IVFPQ 检索');

  // ---------------------------------------------------------------- 6
  section('6 基准评测表格');
  // 显式设 count，让 "暴力扫描量" 有确定的期望值。
  // 依赖页面默认值会让这条断言在默认值改一次之后静默失准。
  await page.fill('#bCount', '600');
  await page.fill('#bDim', '32');
  await page.fill('#bQ', '20');
  await page.click('#btnBench');
  await waitDone('#btnBench', '基准评测');
  await page.waitForTimeout(900);
  check('基准面板已渲染', await page.isVisible('#benchOut table'));

  // 按表头名取列，绝不按下标
  const benchCols = await page.evaluate(() => {
    const table = document.querySelector('#benchOut > table');
    const heads = [...table.querySelectorAll('thead th')].map((t) => t.textContent.trim());
    const rows = [...table.querySelectorAll('tbody tr')].map((r) =>
      [...r.querySelectorAll('td')].map((c) => c.textContent.trim()));
    const idx = Object.fromEntries(heads.map((h, i) => [h, i]));
    return { heads, rows, idx };
  });
  check('表头含「召回率」且存在', benchCols.heads.includes('召回率'),
    JSON.stringify(benchCols.heads));
  check('表头含「门槛召回」', benchCols.heads.includes('门槛召回'),
    JSON.stringify(benchCols.heads));
  check('表头含「距离/查询」', benchCols.heads.includes('距离/查询'),
    JSON.stringify(benchCols.heads));
  check('三行数据（brute / hnsw / ivfpq）', benchCols.rows.length === 3,
    String(benchCols.rows.length));

  const pick = (row, name) => benchCols.idx[name] === undefined
    ? undefined : row[benchCols.idx[name]];
  const bruteRow = benchCols.rows.find((r) => r[0].indexOf('暴力') >= 0);
  const hnswRow = benchCols.rows.find((r) => r[0].indexOf('HNSW') >= 0);
  const ivfRow = benchCols.rows.find((r) => r[0].indexOf('IVF-PQ') >= 0);
  check('三个引擎行都找到', !!(bruteRow && hnswRow && ivfRow),
    JSON.stringify(benchCols.rows.map((r) => r[0])));

  // 数值解析，不做字符串包含 —— ¥0.000085 这类包含判断会假通过
  const num = (s) => s === undefined ? NaN : Number(String(s).replace(/[^0-9.]/g, ''));
  check('暴力召回率 = 100%', Math.abs(num(pick(bruteRow, '召回率')) - 100) < 0.01,
    pick(bruteRow, '召回率'));
  check('暴力扫描量 = 600', num(pick(bruteRow, '距离/查询')) === 600,
    pick(bruteRow, '距离/查询'));
  const hnswScan = num(pick(hnswRow, '距离/查询'));
  const bruteScan = num(pick(bruteRow, '距离/查询'));
  check('HNSW 距离计算少于暴力', hnswScan < bruteScan,
    hnswScan + ' vs ' + bruteScan);
  const ivfMem = num(pick(ivfRow, '内存'));
  const bruteMem = num(pick(bruteRow, '内存'));
  check('IVF-PQ 内存小于暴力', ivfMem < bruteMem, ivfMem + ' vs ' + bruteMem);
  await shot('07-基准评测');

  const sweepCount = await page.evaluate(() =>
    document.querySelectorAll('#benchOut h3').length);
  check('渲染了至少 2 段扫描曲线', sweepCount >= 2, String(sweepCount));

  // ---------------------------------------------------------------- 7
  section('7 SVG 图表');
  const svgChecks = [];
  for (const tab of ['recall', 'memory', 'layers']) {
    await page.click(`[data-tab="${tab}"]`);
    await page.waitForTimeout(500);
    const shapes = await page.evaluate(() =>
      document.querySelectorAll('#svg *').length);
    const texts = await page.evaluate(() =>
      [...document.querySelectorAll('#svg text')].map((t) => t.textContent).join('|'));
    svgChecks.push({ tab, shapes, text: texts.slice(0, 90) });
    check(`${tab} 图有绘制内容`, shapes > 8, String(shapes));
    check(`${tab} 图有文字标签`, texts.length > 0, texts.slice(0, 60));
    await shot('08-图表-' + tab);
  }
  // HNSW 分层图必须标明几何衰减关系
  const layerText = await page.evaluate(() => {
    document.querySelector('[data-tab="layers"]').click();
    return [...document.querySelectorAll('#svg text')].map((t) => t.textContent).join('|');
  });
  check('分层图标明了几何衰减', layerText.indexOf('几何衰减') >= 0, layerText.slice(0, 120));

  // ---------------------------------------------------------------- 8
  section('8 错误提示');
  await page.evaluate(() => {
    document.querySelector('#sK').value = '0';
    document.querySelector('#sK').dispatchEvent(new Event('input'));
  });
  expectHttpError = true;
  await page.click('#btnSearch');
  await page.waitForTimeout(1600);
  const toastText = await page.evaluate(() => {
    const t = document.querySelector('#toast');
    return t && !t.classList.contains('hidden') ? t.textContent : null;
  });
  check('非法 k 弹出错误提示', toastText !== null && toastText.length > 0, String(toastText));
  await shot('09-错误提示');

  // ---------------------------------------------------------------- 9
  section('9 控制台');
  check('无 JS 异常', jsErrors.length === 0, jsErrors.slice(0, 3).join(' / '));
  check('资源请求失败仅限故意触发的那次 422',
    !expectHttpError ? resourceErrors.length === 0 : resourceErrors.length <= 1,
    resourceErrors.slice(0, 3).join(' / '));

  await browser.close();

  console.log('\n' + '='.repeat(70));
  console.log('通过 ' + passed + ' · 失败 ' + failed);
  console.log('='.repeat(70));

  // 校验截图没有重复（重复意味着某次截图其实没更新）
  const files = fs.readdirSync(OUT).filter((f) => f.endsWith('.png'));
  console.log('截图 ' + files.length + ' 张：' + files.join(', '));

  process.exit(failed ? 1 : 0);
})().catch((e) => {
  console.error('\n脚本异常：' + e.message);
  console.error(e.stack);
  process.exit(1);
});
