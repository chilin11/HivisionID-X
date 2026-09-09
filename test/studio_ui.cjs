// Run: NODE_PATH=/path/to/node_modules node test/studio_ui.cjs
// Uses Playwright and mocked API responses; no model weights are needed.
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const png = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII=';
(async () => {
  const browser = await chromium.launch({headless:true});
  const page = await browser.newPage({viewport:{width:1440,height:1050}});
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  let fail = false, watermarks = 0;
  await page.route('http://studio.test/**', async route => {
    const url = new URL(route.request().url());
    if (url.pathname === '/') return route.fulfill({contentType:'text/html',body:fs.readFileSync(path.join(root,'web-ui/dist/index.html'),'utf8')});
    if (url.pathname === '/static/studio.css') return route.fulfill({contentType:'text/css',body:fs.readFileSync(path.join(root,'web-ui/dist/studio.css'),'utf8')});
    let json = {status:true};
    if (url.pathname === '/health') json.super_res = true;
    if (url.pathname === '/idphoto') json = fail ? {status:false,error:'Test failure'} : {status:true,image_base64_hd:png,image_base64_standard:png,image_base64_matting:png,standard_format:'png',hd_format:'png',size:{height:413,width:295},hd_size:{height:1600,width:1143},dpi:300,elapsed_ms:120,quality:{engine:'smart',top_gap:.1,bottom_gap:0,h_center_offset:0},warnings:[]};
    if (url.pathname === '/generate_layout_photos' || url.pathname === '/watermark') { json.image_base64 = png; if(url.pathname === '/watermark') watermarks++; }
    await route.fulfill({json});
  });
  await page.goto('http://studio.test');
  assert.equal(await page.title(),'HivisionID-X · 智能证件照');
  assert.deepEqual(await page.evaluate(() => {const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);return ids.filter((x,i)=>ids.indexOf(x)!==i)}),[]);
  await page.screenshot({path:'/tmp/hivision-studio-desktop.png',fullPage:true});
  await page.locator('#cSzMode [data-v=custom_mm]').click();
  assert.deepEqual(await page.evaluate(() => sz()), {h:413,w:295});
  await page.locator('#cSzMode [data-v=preset]').click();
  await page.locator('#fi').setInputFiles({name:'portrait<test>.png',mimeType:'image/png',buffer:Buffer.from(png.split(',')[1],'base64')});
  await page.locator('#goBtn').click();
  await page.locator('.lay-card').waitFor();
  await page.locator('[data-tab=standard]').click();
  assert.equal(await page.locator('.lay-card').count(),1);
  await page.locator('#resultPreview').click();
  assert.equal(await page.locator('#pixelPreview').evaluate(e=>e.open),true);
  await page.keyboard.press('Escape');
  await page.locator('#xWm .col-h').click();
  await page.locator('#cWm [data-v=on]').click();
  await page.locator('#goBtn').click();
  await page.waitForFunction(() => !S.busy);
  assert.equal(watermarks,2);
  fail = true;
  await page.locator('#forceBtn').click();
  await page.waitForFunction(() => !S.busy);
  assert.equal(await page.locator('.proc').count(),0);
  assert.equal(await page.locator('#goBtn').isEnabled(),true);
  await page.locator('#langBtn').click();
  assert.equal(await page.locator('.studio-intro h1').textContent(),'A better portrait. Every time.');
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:'/tmp/hivision-studio-mobile.png',fullPage:true});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  assert.deepEqual(errors,[]);
  await browser.close();
  console.log('PASS: desktop/mobile, unique IDs, mm size, upload, generation, tabs, pixel preview, watermark, failure recovery, language, no JS errors');
})().catch(e=>{ console.error(e); process.exit(1); });
