// Run: NODE_PATH=$(npm root -g) node test/studio_ui.cjs
// Uses Playwright and mocked model responses; no model weights are needed.
// Optional LAYOUT_API_URL=http://127.0.0.1:PORT verifies real layout encoding.
// Set PLAYWRIGHT_CHROMIUM_CHANNEL=chrome to use an installed Chrome.
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const artifacts = process.env.PI_SCRATCH_DIR || path.join(require('node:os').tmpdir(),'hivision-studio-tests');
fs.mkdirSync(artifacts,{recursive:true});
const png = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII=';
(async () => {
  const browser = await chromium.launch({headless:true,...(process.env.PLAYWRIGHT_CHROMIUM_CHANNEL ? {channel:process.env.PLAYWRIGHT_CHROMIUM_CHANNEL} : {})});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1050}});
    const hd = await page.evaluate(() => {
      const canvas = document.createElement('canvas'); canvas.width=1143; canvas.height=1600;
      const ctx = canvas.getContext('2d');
      const gradient = ctx.createLinearGradient(0,0,1143,1600); gradient.addColorStop(0,'#438edb'); gradient.addColorStop(1,'#e4cab1');
      ctx.fillStyle=gradient; ctx.fillRect(0,0,1143,1600);
      for(let x=0;x<1143;x+=7) { ctx.fillStyle=x%2?'#284e31':'#fff'; ctx.fillRect(x,400,1,800); }
      return canvas.toDataURL('image/png');
    });
    const errors = [], layouts = [];
    page.on('pageerror', e => errors.push(e.message));
    let fail = false, watermarks = 0;
    await page.route('http://studio.test/**', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/') return route.fulfill({contentType:'text/html',body:fs.readFileSync(path.join(root,'web-ui/dist/index.html'),'utf8')});
      if (url.pathname === '/static/studio.css') return route.fulfill({contentType:'text/css',body:fs.readFileSync(path.join(root,'web-ui/dist/studio.css'),'utf8')});
      let json = {status:true};
      if (url.pathname === '/health') json.super_res = true;
      if (url.pathname === '/idphoto') json = fail ? {status:false,error:'Test failure'} : {status:true,image_base64_hd:hd,image_base64_standard:png,image_base64_matting:png,standard_format:'png',hd_format:'png',size:{height:413,width:295},hd_size:{height:1600,width:1143},dpi:300,elapsed_ms:120,quality:{engine:'smart',top_gap:.1,bottom_gap:0,h_center_offset:0},warnings:[]};
      if (url.pathname === '/generate_layout_photos') {
        const body = route.request().postDataBuffer();
        const text = body.toString('latin1');
        const field = name => text.match(new RegExp('name="'+name+'"\\r\\n\\r\\n([^\\r]+)'))?.[1];
        const fields = Object.fromEntries(['height','width','dpi','output_dpi','layout_height','layout_width','kb','crop_line'].map(name=>[name,field(name)]));
        fields.hdSource = body.includes(Buffer.from(hd.split(',')[1],'base64'));
        layouts.push(fields);
        if (process.env.LAYOUT_API_URL) {
          const response = await route.fetch({url:process.env.LAYOUT_API_URL+'/generate_layout_photos'});
          return route.fulfill({response});
        }
        json = {status:true,image_base64:hd,size:{height:+fields.layout_height*2,width:+fields.layout_width*2},dpi:600};
      }
      if (url.pathname === '/watermark') { json.image_base64 = hd; watermarks++; }
      await route.fulfill({json});
    });
    await page.goto('http://studio.test');
    assert.equal(await page.title(),'HivisionID-X · 智能证件照');
    assert.deepEqual(await page.evaluate(() => {const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);return ids.filter((x,i)=>ids.indexOf(x)!==i)}),[]);
    await page.locator('#cSzMode [data-v=custom_mm]').click();
    assert.deepEqual(await page.evaluate(() => sz()), {h:413,w:295});
    await page.locator('#cSzMode [data-v=preset]').click();
    await page.locator('#fi').setInputFiles({name:'portrait<test>.png',mimeType:'image/png',buffer:Buffer.from(png.split(',')[1],'base64')});
    await page.locator('#goBtn').click();
    await page.locator('.lay-card').waitFor();
    const layoutName = () => page.locator('.lay-card a[download]').getAttribute('download');
    assert.equal(await layoutName(),'layout_2410x3590_600dpi.png');
    assert.equal(layouts.at(-1).hdSource,true);
    assert.equal(layouts.at(-1).output_dpi,'600');
    assert.equal(layouts.at(-1).dpi,'300');
    assert.equal(layouts.at(-1).kb,undefined);
    assert.match(await page.locator('.lay-bar').textContent(),/3590×2410 px · 600 DPI · PNG/);
    assert.match(await page.locator('.lay-card').textContent(),/100%/);
    if (process.env.LAYOUT_API_URL) {
      await page.locator('.lay-card img').evaluate(img => img.decode());
      assert.deepEqual(await page.locator('.lay-card img').evaluate(img=>[img.naturalWidth,img.naturalHeight]),[3590,2410]);
      const downloadPromise = page.waitForEvent('download');
      await page.locator('.lay-card a[download]').click();
      const download = await downloadPromise;
      await download.saveAs(path.join(artifacts,download.suggestedFilename()));
    }
    await page.locator('[data-tab=standard]').click();
    assert.equal(await page.locator('.lay-card').count(),1);
    await page.locator('#resultPreview').click();
    assert.equal(await page.locator('#pixelPreview').evaluate(e=>e.open),true);
    await page.keyboard.press('Escape');
    await page.locator('#cPr [data-h="1051"]').first().click();
    await page.waitForFunction(()=>document.querySelector('.lay-card a[download]')?.download==='layout_2102x3000_600dpi.png');
    await page.locator('#cPr [data-w="3508"]').click();
    await page.waitForFunction(()=>document.querySelector('.lay-card a[download]')?.download==='layout_4958x7016_600dpi.png');
    assert.equal(await layoutName(),'layout_4958x7016_600dpi.png');
    await page.locator('#cPr [data-h="1205"]').first().click();
    await page.waitForFunction(()=>document.querySelector('.lay-card a[download]')?.download==='layout_2410x3590_600dpi.png');
    await page.evaluate(async()=>{
      S.result={...S.result,size:{height:826,width:590},dpi:600};
      S.kb=20; S.plugs.jpeg=true;
      await genLayout(true);
      S.kb=null; S.plugs.jpeg=false;
    });
    assert.equal(layouts.at(-1).height,'413');
    assert.equal(layouts.at(-1).width,'295');
    assert.equal(layouts.at(-1).dpi,'300');
    assert.equal(layouts.at(-1).kb,undefined);
    await page.screenshot({path:path.join(artifacts,'hivision-studio-desktop.png'),fullPage:true});
    await page.locator('#xWm .col-h').click();
    await page.locator('#cWm [data-v=on]').click();
    await page.locator('#goBtn').click();
    await page.waitForFunction(() => !S.busy);
    await page.locator('.lay-card').waitFor();
    assert.equal(watermarks,2);
    fail = true;
    await page.locator('#forceBtn').click();
    await page.waitForFunction(() => !S.busy);
    assert.equal(await page.locator('.proc').count(),0);
    assert.equal(await page.locator('#goBtn').isEnabled(),true);
    await page.locator('#langBtn').click();
    assert.equal(await page.locator('.studio-intro h1').textContent(),'A better portrait. Every time.');
    assert.equal(await page.locator('#xPr [data-i18n=layoutQuality]').textContent(),'600 DPI · HD source · Lossless PNG');
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(artifacts,'hivision-studio-mobile.png'),fullPage:true});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    assert.deepEqual(errors,[]);
    console.log('PASS: HD-source/600-DPI layouts, six/five-inch/A4, custom DPI normalization, no KB limit, actual filenames, desktop/mobile, tabs, watermark, failure recovery, language, no JS errors'+(process.env.LAYOUT_API_URL?', real layout API + PNG download':''));
  } finally { await browser.close(); }
})().catch(e=>{ console.error(e); process.exit(1); });
