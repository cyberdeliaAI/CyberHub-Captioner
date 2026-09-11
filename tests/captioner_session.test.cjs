// Run with: node --test tests/captioner_session.test.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const modulePath = process.env.CAPTIONER_SOURCE || path.join(__dirname, '../modules/captioner/__init__.py');
// PAGE_BODY is a raw Python string; its inline JavaScript needs no unescaping.
const source = fs.readFileSync(modulePath, 'utf8').split('<script>')[1].split('</script>')[0].replace(/init\(\);\s*$/, '');

function harness() {
  class Element {
    constructor() { this.value = ''; this.style = {}; this.dataset = {}; this.listeners = {}; this.children = []; this.classList = {add() {}, remove() {}, toggle() {}}; }
    get options() { return Array.from((this.innerHTML || '').matchAll(/<option value="([^"]*)"/g), match => ({value:match[1]})); }
    addEventListener(name, fn) { this.listeners[name] = fn; }
    appendChild(child) { this.children.push(child); }
    setAttribute() {}
    focus() {}
    reportValidity() { return true; }
    close() { this.open = false; this.listeners.close?.(); }
    showModal() { this.open = true; }
    getBoundingClientRect() { return {left: 0, right: 600, top: 0, bottom: 700}; }
  }
  const elements = new Map();
  const el = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  const context = vm.createContext({
    console: {...console, warn() {}}, Blob, File, URL, TextEncoder, AbortController, AbortSignal,
    setTimeout: (fn, delay) => delay === 0 ? setTimeout(fn, 0) : 1, clearTimeout() {}, setInterval() {},
    document: {getElementById: el, createElement: () => new Element(), querySelectorAll: () => [], addEventListener() {}},
    window: {addEventListener() {}}, navigator: {},
    localStorage: {getItem: () => null, setItem() {}},
  });
  vm.runInContext(source, context);
  const run = code => vm.runInContext(code, context);
  run(`toast = () => {}; state.saveSidecars = false; state.config = {api_url:'http://saved.test/v1', model:'saved-model', temperature:null};`);
  const state = run('state');
  const img = {id: 1, name: 'sample.png', file: new File(['test'], 'sample.png', {type:'image/png'}), caption:'Generated caption', status:'done', sidecarStatus:'saved', url:'blob:test'};
  state.images = [img]; state.activeIdx = 0;
  const edit = text => { el('captionOutput').value = text; el('captionOutput').listeners.input(); };
  return {run, state, img, el, edit, context};
}

test('manual edits survive image switching and appear in the session record', async () => {
  const h = harness();
  h.edit('Hand-edited caption with café and a new detail.');
  assert.equal(h.img.sidecarStatus, 'pending');
  h.run(`state.images.push({id:2,name:'second.png',caption:'Second caption',status:'done',url:'blob:second'}); selectImage(1); selectImage(0);`);
  assert.equal(h.el('captionOutput').value, h.img.caption);
  h.run(`sessionStore = async (_mode, fn) => fn({put(record) { globalThis.savedRecord = record; }});`);
  await h.run('saveSessionNow()');
  assert.equal(h.run('savedRecord.images[0].caption'), h.img.caption);
  assert.equal(h.run('sessionSaveState'), 'saved');
});

function share(overrides={}) {
  return {schema:1, product:'CyberHub', type:'system_prompt', module:'captioner', title:'A shared prompt', prompt:'Describe this image.\nBegin with {TRIGGER}.', ...overrides};
}

test('shared JSON round-trips the edited title, prompt, Unicode and placeholders only', () => {
  const h = harness();
  h.el('presetTitle').value = 'Café portrait';
  h.el('presetEditor').value = '  Caption café lights.\nBegin with {TRIGGER}.\n';
  h.run(`download = (filename, text, type) => {globalThis.exported = {filename,text,type};}; exportPromptJson();`);
  const exported = h.run('exported');
  assert.equal(exported.filename, 'cafe-portrait.captioner.json');
  const data = JSON.parse(exported.text);
  assert.deepEqual(Object.keys(data).sort(), ['module','product','prompt','schema','title','type']);
  assert.equal(data.prompt, h.el('presetEditor').value);
  h.context.sharedDocument = data;
  h.run('applyImportedPrompt(sharedDocument)');
  assert.equal(h.el('presetTitle').value, data.title);
  assert.equal(h.el('presetEditor').value, data.prompt);
  assert.equal(h.el('presetSelect').value, 'imported:current');
  assert.equal(h.el('updatePresetBtn').style.display, 'none');
});

test('other modules, future schemas and malformed payloads leave the current prompt intact', () => {
  const h = harness();
  for (const bad of [null, [], {title:'Bare prompt',prompt:'Text'}, share({module:'prompt_engineer'}), share({schema:2}), share({schema:'1'}), share({title:''}), share({title:'x'.repeat(201)}), share({prompt:' '}), share({prompt:42}), share({prompt:'x'.repeat(100001)})]) {
    h.context.sharedDocument = bad;
    assert.throws(() => h.run('applyImportedPrompt(sharedDocument)'));
    assert.equal(h.state.importedPreset, null);
    assert.equal(h.state.activeOverride, '');
  }
});

test('JSON input handles a UTF-8 BOM, repeated file selection and invalid JSON', async () => {
  const h = harness();
  h.context.fileInputForTest = {files:[{size:200, text:async () => '\uFEFF' + JSON.stringify(share())}], value:'selected.json'};
  await h.run('importPromptFile(fileInputForTest)');
  assert.equal(h.state.activeOverride, share().prompt);
  assert.equal(h.context.fileInputForTest.value, '');
  h.context.fileInputForTest.files = [{size:4, text:async () => '{bad'}];
  await h.run('importPromptFile(fileInputForTest)');
  assert.equal(h.state.activeOverride, share().prompt);
  assert.match(h.el('promptTransferStatus').textContent, /not valid JSON/);
  assert.equal(h.el('importPromptBtn').disabled, false);
});

test('oversized files are rejected without reading their contents', async () => {
  const h = harness();
  h.context.fileInputForTest = {files:[{size:512*1024+1, text:async () => {throw new Error('Should not read');}}], value:'large.json'};
  await h.run('importPromptFile(fileInputForTest)');
  assert.match(h.el('promptTransferStatus').textContent, /512 KB/);
  assert.equal(h.state.importedPreset, null);
});

test('an imported prompt and its edits survive a session with no images', async () => {
  const h = harness(); h.state.images = []; h.state.activeIdx = null;
  h.context.sharedDocument = share({title:'Portrait draft'});
  h.run('applyImportedPrompt(sharedDocument)');
  h.el('presetTitle').value = 'My revised title';
  h.el('presetEditor').value = 'Revised text with {TRIGGER}';
  h.run(`state.activeOverride = document.getElementById('presetEditor').value; syncImportedPrompt(); sessionStore = async (_mode, fn) => fn({put(record) {globalThis.savedRecord = structuredClone(record);}});`);
  h.context.structuredClone = structuredClone;
  await h.run('saveSessionNow()');
  h.run(`state.importedPreset = null; state.activePreset = null; state.activeOverride = ''; sessionStore = async (_mode, fn) => fn({get() {return savedRecord;}});`);
  await h.run('restoreSession()');
  assert.equal(h.state.images.length, 0);
  assert.equal(h.el('presetSelect').value, 'imported:current');
  assert.equal(h.el('presetTitle').value, 'My revised title');
  assert.equal(h.el('presetEditor').value, 'Revised text with {TRIGGER}');
});

test('imported titles are escaped in dropdown markup and prompt text stays literal', () => {
  const h = harness();
  h.context.sharedDocument = share({title:'<img src=x onerror=alert(1)>', prompt:'<script>alert(1)</script>\n{TRIGGER}'});
  h.run('applyImportedPrompt(sharedDocument)');
  assert.ok(!h.el('presetSelect').innerHTML.includes('<img'));
  assert.match(h.el('presetSelect').innerHTML, /&lt;img/);
  assert.equal(h.el('presetEditor').value, h.context.sharedDocument.prompt);
});

test('built-in scripts and imported prompts work without the optional Library module', async () => {
  const h = harness();
  h.run('fetch = async () => ({ok:false,status:404})');
  await h.run('loadPresets()');
  assert.equal(h.state.activePreset.id, 'built:z_image');
  h.context.sharedDocument = share();
  h.run('applyImportedPrompt(sharedDocument)');
  await h.run('loadPresets()');
  assert.equal(h.el('presetSelect').value, 'imported:current');
  assert.equal(h.el('presetEditor').value, share().prompt);
});

test('saving an imported preset creates a Captioner Library card with its title and prompt', async () => {
  const h = harness();
  h.context.sharedDocument = share();
  h.run(`applyImportedPrompt(sharedDocument); fetch = async (url, options) => {globalThis.libraryRequest = {url,body:JSON.parse(options.body)}; return {ok:true,json:async () => ({ok:true,id:12,preset:{id:12,title:libraryRequest.body.title,content:libraryRequest.body.content,type:'captioner'}})};}`);
  await h.run('saveAsPreset()');
  assert.equal(h.run('libraryRequest.url'), '/api/captioner/presets/save');
  assert.equal(h.run('libraryRequest.body.title'), share().title);
  assert.equal(h.run('libraryRequest.body.content'), share().prompt);
  assert.equal(h.el('presetSelect').value, 'saved:12');
  assert.equal(h.el('deletePresetBtn').style.display, '');
});

test('double-clicking Save issues only one write and releases controls afterwards', async () => {
  const h = harness();
  h.context.sharedDocument = share();
  let writes = 0, release;
  h.context.fetch = () => {writes++; return new Promise(resolve => {release=resolve;});};
  h.run('applyImportedPrompt(sharedDocument)');
  const first = h.run('saveAsPreset()');
  await h.run('saveAsPreset()');
  assert.equal(writes, 1);
  assert.equal(h.el('savePresetBtn').disabled, true);
  release({ok:true,json:async () => ({ok:true,id:10,preset:{id:10,title:share().title,content:share().prompt,type:'captioner'}})});
  await first;
  assert.equal(h.el('savePresetBtn').disabled, false);
  assert.equal(h.el('presetSelect').value, 'saved:10');
});

test('name conflicts keep the current editor text and show the server explanation', async () => {
  const h = harness();
  h.context.sharedDocument = share();
  h.run(`applyImportedPrompt(sharedDocument); fetch = async () => ({ok:false,json:async () => ({error:'A saved Captioner preset with this name already exists. Choose another title or use Update.'})});`);
  await h.run('saveAsPreset()');
  assert.equal(h.el('presetEditor').value, share().prompt);
  assert.equal(h.el('presetTitle').value, share().title);
  assert.match(h.el('promptTransferStatus').textContent, /already exists/);
  assert.equal(h.state.presets.length, 0);
});

test('delete removes only the selected saved preset and resets the editor selection', async () => {
  const h = harness();
  h.run(`state.presets=[{id:10,title:'Same name',content:'First'},{id:11,title:'Same name',content:'Second'}]; renderPresetOptions('saved:11'); onPresetChange(); confirm=()=>true; fetch=async(url,options)=>{globalThis.deleteRequest={url,body:JSON.parse(options.body)}; return {ok:true,json:async()=>({ok:true,id:11})};};`);
  assert.match(h.el('presetSelect').innerHTML, /Same name · #10/);
  assert.match(h.el('presetSelect').innerHTML, /Same name · #11/);
  await h.run('deletePreset()');
  assert.equal(h.run('deleteRequest.url'), '/api/captioner/presets/delete');
  assert.equal(h.run('deleteRequest.body.id'), 11);
  assert.equal(h.state.presets.length, 1);
  assert.equal(h.state.presets[0].id, 10);
  assert.equal(h.el('presetSelect').value, 'built:z_image');
  assert.equal(h.el('deletePresetBtn').style.display, 'none');
});

test('cancelled deletion and built-in presets never send a delete request', async () => {
  const h = harness();
  h.run(`state.presets=[{id:10,title:'Keep',content:'Text'}]; renderPresetOptions('saved:10'); onPresetChange(); confirm=()=>false; fetch=async()=>{throw new Error('Must not call');};`);
  await h.run('deletePreset()');
  assert.equal(h.state.presets.length, 1);
  h.run(`renderPresetOptions('built:z_image'); onPresetChange(); confirm=()=>true;`);
  await h.run('deletePreset()');
  assert.equal(h.state.presets.length, 1);
});

test('failed deletion keeps the preset and editor unchanged', async () => {
  const h = harness();
  h.run(`state.presets=[{id:10,title:'Keep',content:'Text'}]; renderPresetOptions('saved:10'); onPresetChange(); confirm=()=>true; fetch=async()=>({ok:false,json:async()=>({error:'Library unavailable'})});`);
  await h.run('deletePreset()');
  assert.equal(h.state.presets.length, 1);
  assert.equal(h.el('presetEditor').value, 'Text');
  assert.match(h.el('promptTransferStatus').textContent, /Library unavailable/);
  assert.equal(h.el('deletePresetBtn').disabled, false);
});

test('Save caption files passes the edited text to the ZIP exporter', async () => {
  const h = harness();
  h.edit('The exact edited output.');
  h.run(`supportsWritableFolders = () => false; downloadBlob = () => {}; fetch = async (url, options) => { globalThis.exportBody = JSON.parse(options.body); return {ok:true, blob:async () => new Blob(['zip'])}; };`);
  await h.run('saveSidecarsNow()');
  assert.equal(h.run('exportBody.items[0].caption'), 'The exact edited output.');
});

test('direct caption files contain manual edits', async () => {
  const h = harness();
  const writes = [];
  h.state.sidecarDirectory = {getFileHandle: async name => { assert.equal(name, 'sample.txt'); return {createWritable: async () => ({write: async text => writes.push(text), close: async () => {}})}; }};
  h.edit('Updated file text');
  h.run('supportsWritableFolders = () => true');
  await h.run('saveSidecarsNow()');
  assert.deepEqual(writes, ['Updated file text\n']);
  assert.equal(h.img.sidecarStatus, 'saved');
});

test('an edit made while a file is being written stays dirty', async () => {
  const h = harness();
  let release, started;
  const began = new Promise(resolve => { started = resolve; });
  const gate = new Promise(resolve => { release = resolve; });
  const writes = [];
  h.state.sidecarDirectory = {getFileHandle: async () => ({createWritable: async () => ({write: async text => {writes.push(text); if (writes.length === 1) {started(); await gate;}}, close: async () => {}})})};
  const first = h.run('writeSidecar(state.images[0])');
  await began;
  h.edit('Latest edit');
  release(); await first;
  assert.equal(h.img.sidecarStatus, 'pending');
  await h.run('writeSidecar(state.images[0])');
  assert.deepEqual(writes, ['Generated caption\n', 'Latest edit\n']);
  assert.equal(h.img.sidecarStatus, 'saved');
});

test('overlapping blur and manual saves never open two writers for one image', async () => {
  const h = harness();
  let active = 0, maxActive = 0, release, started;
  const writes = [];
  const began = new Promise(resolve => {started = resolve;});
  const gate = new Promise(resolve => {release = resolve;});
  h.state.sidecarDirectory = {getFileHandle: async () => ({createWritable: async () => {
    maxActive = Math.max(maxActive, ++active);
    return {write: async text => {writes.push(text); if (writes.length === 1) {started(); await gate;}}, close: async () => {active--;}};
  }})};
  const first = h.run('writeSidecar(state.images[0])');
  await began;
  h.edit('Latest concurrent edit');
  const second = h.run('writeSidecar(state.images[0])');
  release(); await Promise.all([first, second]);
  assert.equal(maxActive, 1);
  assert.deepEqual(writes, ['Generated caption\n', 'Latest concurrent edit\n']);
  assert.equal(h.img.sidecarStatus, 'saved');
});

test('generation completing after an edit preserves the manual text everywhere', async () => {
  const h = harness();
  let started, complete;
  const began = new Promise(resolve => {started = resolve;});
  h.context.generate = () => {started(); return new Promise(resolve => {complete = resolve;});};
  h.run(`fileToBase64 = async () => 'test'; captionViaBrowser = generate;`);
  const running = h.run('captionImage(state.images[0])');
  await began;
  h.edit('Keep my manual edit');
  complete('Late model response'); await running;
  assert.equal(h.img.caption, 'Keep my manual edit');
  assert.equal(h.el('captionOutput').value, 'Keep my manual edit');
  assert.equal(h.img.sidecarStatus, 'pending');
});

test('a focused editor without edits receives the generated caption', async () => {
  const h = harness();
  h.context.document.activeElement = h.el('captionOutput');
  h.run(`fileToBase64 = async () => 'test'; captionViaBrowser = async () => 'New model result';`);
  await h.run('captionImage(state.images[0])');
  assert.equal(h.el('captionOutput').value, 'New model result');
  assert.equal(h.img.caption, 'New model result');
});

test('manually captioned pending images update progress and are skipped by Run uncaptioned', () => {
  const h = harness(); h.img.status = 'pending'; h.img.caption = '';
  h.edit('Written by hand');
  assert.equal(h.img.status, 'done');
  assert.equal(h.el('progressText').textContent, '1 / 1 captioned');
  assert.equal(h.el('runUncaptionedBtn').disabled, true);
  h.edit('');
  assert.equal(h.img.status, 'pending');
  assert.equal(h.el('runUncaptionedBtn').disabled, false);
});

test('browser storage failure is visible to the user', async () => {
  const h = harness(); h.edit('Keep this text');
  h.run(`sessionStore = async () => {throw new Error('Storage quota');};`);
  await h.run('saveSessionNow()');
  assert.equal(h.run('sessionSaveState'), 'error');
  assert.match(h.el('captionSaveStatus').textContent, /Browser save failed/);
});

test('unsaved connection fields cannot change generation requests', () => {
  const h = harness();
  h.el('capApiUrl').value = 'http://draft.test';
  h.el('capModel').value = 'draft-model';
  const request = h.run(`buildDirectCaptionPayload('test','image/png','Describe')`);
  assert.equal(request.cfg.api_url, 'http://saved.test/v1');
  assert.equal(request.payload.model, 'saved-model');
  assert.equal('temperature' in request.payload, false);
  h.run('setupConnectionDialog(); openConnectionSettings();');
  h.el('capModel').value = 'discard';
  h.run('closeConnectionSettings()');
  assert.equal(h.el('capModel').value, 'saved-model');
});

test('failed settings save keeps the dialog and saved configuration intact', async () => {
  const h = harness();
  h.run(`setupConnectionDialog(); openConnectionSettings(); fetch = async () => ({ok:false, json:async () => ({error:'Server unavailable'})});`);
  h.el('capModel').value = 'new-model';
  assert.equal(await h.run('saveConnectionConfig()'), false);
  assert.equal(h.state.config.model, 'saved-model');
  assert.equal(h.el('connectionDialog').open, true);
  assert.match(h.el('connectionFeedback').textContent, /Server unavailable/);
  assert.equal(h.state.connectionBusy, false);
});

// Validate generated archives with an independent ZIP reader, including CRCs.
async function readDatasetArchive(blob) {
  const {spawnSync} = require('node:child_process');
  const python = `import base64, io, json, sys, zipfile
with zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read())) as archive:
    assert archive.testzip() is None
    print(json.dumps([{'name': i.filename, 'data': base64.b64encode(archive.read(i)).decode(), 'method': i.compress_type, 'flags': i.flag_bits} for i in archive.infolist()]))
`;
  const result = spawnSync('python3', ['-B', '-c', python], {input:Buffer.from(await blob.arrayBuffer()), maxBuffer:20*1024*1024});
  assert.equal(result.status, 0, result.stderr.toString());
  return JSON.parse(result.stdout.toString()).map(entry => ({...entry, bytes:Buffer.from(entry.data, 'base64')}));
}

function captureDatasetDownload(h) {
  let download;
  h.context.downloadBlob = (filename, blob) => { download = {filename, blob}; };
  h.context.toast = text => { h.context.lastToast = text; };
  h.context.fetch = () => { throw new Error('Dataset export must not upload images'); };
  h.context.window.showDirectoryPicker = () => { throw new Error('Dataset export must not request folder access'); };
  return () => download;
}

test('dataset ZIP contains original image bytes and matching edited captions, skipping empty captions', async () => {
  const h = harness(), downloaded = captureDatasetDownload(h);
  const original = Uint8Array.from([0, 255, 2, 128, 10, 0, 239]);
  h.img.file = new File([original], 'café.png', {type:'image/png'});
  h.img.name = 'café.png';
  h.edit('  My café caption.\nA second line.  ');
  h.state.images.push({name:'café.jpg', file:new File(['other original'], 'café.jpg'), caption:'Second image'},
    {name:'pending.png', file:new File(['not included'], 'pending.png'), caption:'  '});
  h.context.supportsWritableFolders = () => true;
  await h.run('exportDatasetZip()');
  assert.equal(downloaded().filename, 'images-and-captions.zip');
  assert.equal(downloaded().blob.type, 'application/zip');
  const entries = await readDatasetArchive(downloaded().blob);
  assert.deepEqual(entries.map(e => e.name), ['café.png','café.txt','café_2.jpg','café_2.txt']);
  assert.deepEqual(entries[0].bytes, Buffer.from(original));
  assert.equal(entries[1].bytes.toString(), 'My café caption.\nA second line.\n');
  assert.equal(entries[2].bytes.toString(), 'other original');
  assert.equal(entries[3].bytes.toString(), 'Second image\n');
  assert.ok(entries.every(e => e.method === 0 && e.flags & 0x0800));
  assert.equal(h.img.sidecarStatus, 'pending');
});

test('dataset names remain safe and pairs stay unique across folders, case and existing suffixes', async () => {
  const h = harness();
  h.state.images = ['../photo.png','folder\\PHOTO.jpg','photo_2.webp','CON.png','..','misnamed.txt','café.png','cafe\u0301.jpg'].map(name => ({name, file:new File(['original'],name),caption:'Caption'}));
  const entries = await readDatasetArchive(await h.run('buildDatasetZip(datasetZipEntries(state.images))'));
  const names = entries.map(e => e.name);
  assert.equal(new Set(names.map(name => name.normalize('NFKC').toLowerCase())).size, names.length);
  assert.ok(names.every(name => !/[\\/]/.test(name) && name !== '..'));
  assert.ok(names.includes('_CON.png'));
  for (let index = 0; index < names.length; index += 2) {
    const image = names[index], dot = image.lastIndexOf('.');
    assert.equal(names[index+1], (dot > 0 ? image.slice(0,dot) : image) + '.txt');
  }
});

test('dataset ZIP keeps a consistent snapshot when captions change during export and blocks double clicks', async () => {
  const h = harness(), downloaded = captureDatasetDownload(h);
  let started, release;
  const began = new Promise(resolve => {started=resolve;});
  const gate = new Promise(resolve => {release=resolve;});
  const originalSlice = h.img.file.slice.bind(h.img.file);
  h.img.file.slice = (...args) => ({arrayBuffer: async () => {started(); await gate; return originalSlice(...args).arrayBuffer();}});
  h.edit('Caption at click');
  const first = h.run('exportDatasetZip()');
  await began;
  assert.equal(h.el('downloadDatasetBtn').disabled,true);
  h.edit('Edited while preparing');
  await h.run('exportDatasetZip()');
  assert.equal(downloaded(),undefined);
  release(); await first;
  const entries = await readDatasetArchive(downloaded().blob);
  assert.equal(entries[1].bytes.toString(),'Caption at click\n');
  assert.equal(h.img.caption,'Edited while preparing');
  assert.equal(h.state.datasetExportBusy,false);
  assert.equal(h.el('downloadDatasetBtn').disabled,false);
});

test('dataset export fails without a partial download when any captioned original is missing or unreadable', async () => {
  for (const failure of ['missing','unreadable']) {
    const h = harness(), downloaded = captureDatasetDownload(h);
    if (failure === 'missing') h.img.file = null;
    else h.img.file.slice = () => ({arrayBuffer:async () => {throw new Error('File could not be read');}});
    await h.run('exportDatasetZip()');
    assert.equal(downloaded(),undefined);
    assert.match(h.context.lastToast, failure === 'missing' ? /Original image unavailable/ : /could not be read/);
    assert.equal(h.state.datasetExportBusy,false);
    assert.equal(h.el('downloadDatasetBtn').disabled,false);
  }
});

test('dataset export is disabled without nonempty captions and restores controls on failure', async () => {
  const h = harness(), downloaded = captureDatasetDownload(h);
  h.edit('  ');
  assert.equal(h.el('downloadDatasetBtn').disabled,true);
  await h.run('exportDatasetZip()');
  assert.equal(downloaded(),undefined);
  assert.match(h.context.lastToast,/No captioned images/);
  assert.equal(h.state.datasetExportBusy,false);
  assert.equal(h.el('downloadDatasetBtn').disabled,true);
});

test('oversized datasets are rejected before any image is read', async () => {
  const h = harness();
  h.context.largeEntries = [{name:'large.png',blob:{size:2*1024*1024*1024,slice(){throw new Error('Must not read');}}}];
  await assert.rejects(h.run('buildDatasetZip(largeEntries)'), /2 GB/);
  h.context.tooManyEntries = new Array(20002).fill({name:'image.png',blob:{size:1}});
  await assert.rejects(h.run('buildDatasetZip(tooManyEntries)'), /10,000/);
});

test('ZIP CRCs and sizes remain valid for images spanning multiple read chunks', async () => {
  const h = harness();
  const bytes = Buffer.alloc(1024*1024+123, 0xa7);
  h.img.file = new File([bytes], 'sample.png');
  const entries = await readDatasetArchive(await h.run('buildDatasetZip(datasetZipEntries(state.images))'));
  assert.deepEqual(entries[0].bytes, bytes);
});
