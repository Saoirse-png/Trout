// script.js
// Flow-based eastward "streaky" trout generator with ocean-mask sampling and simple overlays
// Implements: build ocean mask from image luminance, advected particles following a biased flow field, draw trout on canvas, export CSV/JSON/PNG, and simple overlay upload/auto-stamp/manual stamp.

(() => {
  // --- Utilities ---
  function mulberry32(a) {
    return function() {
      let t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
  }

  // 2D value-noise with bilinear interpolation (seeded)
  function makeNoise2D(seed) {
    const rand = mulberry32(seed >>> 0);
    // permutation table
    const perm = new Uint8Array(512);
    const p = new Uint8Array(256);
    for (let i=0;i<256;i++) p[i]=i;
    for (let i=255;i>0;i--) {
      const j = Math.floor(rand()*(i+1));
      const tmp = p[i]; p[i]=p[j]; p[j]=tmp;
    }
    for (let i=0;i<512;i++) perm[i]=p[i & 255];

    function smoothstep(t){ return t*t*(3-2*t); }
    return function(x,y){
      // grid cell
      const X = Math.floor(x) & 255;
      const Y = Math.floor(y) & 255;
      const xf = x - Math.floor(x);
      const yf = y - Math.floor(y);
      const u = smoothstep(xf);
      const v = smoothstep(yf);
      const aa = perm[X + perm[Y]];
      const ab = perm[X + perm[Y + 1]];
      const ba = perm[X + 1 + perm[Y]];
      const bb = perm[X + 1 + perm[Y + 1]];
      // produce pseudo-random gradients from perm values
      const grad = (hash, dx, dy) => {
        // convert hash to pseudo-gradient vector in [-1,1]
        const h = hash & 7;
        const gx = (h & 1) ? 1 : -1;
        const gy = (h & 2) ? 1 : -1;
        return gx*dx + gy*dy;
      };
      const x1 = grad(aa, xf, yf) * (1 - u) + grad(ba, xf-1, yf) * u;
      const x2 = grad(ab, xf, yf-1) * (1 - u) + grad(bb, xf-1, yf-1) * u;
      const value = x1 * (1 - v) + x2 * v;
      // value roughly in range [-2,2] depending on grad; clamp then normalize
      return Math.max(-1, Math.min(1, value));
    }
  }

  // --- DOM ---
  const img = document.getElementById('sourceImage');
  const fileInput = document.getElementById('fileInput');
  const thresholdEl = document.getElementById('threshold');
  const thresholdVal = document.getElementById('thresholdVal');
  const countEl = document.getElementById('count');
  const generateBtn = document.getElementById('generate');
  const clearBtn = document.getElementById('clear');
  const exportCsv = document.getElementById('exportCsv');
  const exportJson = document.getElementById('exportJson');
  const exportPng = document.getElementById('exportPng');

  const flowStrengthEl = document.getElementById('flowStrength');
  const flowVal = document.getElementById('flowVal');
  const noiseScaleEl = document.getElementById('noiseScale');
  const noiseVal = document.getElementById('noiseVal');
  const diffusionEl = document.getElementById('diffusion');
  const diffVal = document.getElementById('diffVal');
  const stepSizeEl = document.getElementById('stepSize');
  const maxStepsEl = document.getElementById('maxSteps');
  const seedEl = document.getElementById('seed');

  const overlaysCanvas = document.getElementById('overlaysCanvas');
  const troutCanvas = document.getElementById('troutCanvas');
  const stampsCanvas = document.getElementById('stampsCanvas');

  const overlayUpload = document.getElementById('overlayUpload');
  const stampSizeEl = document.getElementById('stampSize');
  const stampCountEl = document.getElementById('stampCount');
  const stampOpacityEl = document.getElementById('stampOpacity');
  const stampRotateEl = document.getElementById('stampRotate');
  const restrictOceanEl = document.getElementById('restrictOcean');
  const addOverlayBtn = document.getElementById('addOverlay');
  const overlayList = document.getElementById('overlayList');

  // hidden canvas for pixel reads
  const readCanvas = document.createElement('canvas');
  const readCtx = readCanvas.getContext('2d');

  let oceanPixels = []; // array of {x,y}
  let currentPoints = []; // array of {x,y,streakId,step}

  let overlays = []; // list of layers
  let noise2 = makeNoise2D(1337);
  let noise3 = makeNoise2D(4242);

  // --- helpers ---
  function updateRangeLabels(){
    flowVal.textContent = parseFloat(flowStrengthEl.value).toFixed(2);
    noiseVal.textContent = parseFloat(noiseScaleEl.value).toFixed(2);
    diffVal.textContent = parseFloat(diffusionEl.value).toFixed(2);
    thresholdVal.textContent = thresholdEl.value;
  }
  flowStrengthEl.addEventListener('input', updateRangeLabels);
  noiseScaleEl.addEventListener('input', updateRangeLabels);
  diffusionEl.addEventListener('input', updateRangeLabels);
  thresholdEl.addEventListener('input', updateRangeLabels);
  updateRangeLabels();

  fileInput.addEventListener('change', (ev)=>{
    const f = ev.target.files && ev.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => { img.src = reader.result; };
    reader.readAsDataURL(f);
  });

  // ensure canvases match image natural resolution and styled display size
  function resizeCanvases(){
    if (!img.naturalWidth) return;
    [overlaysCanvas, troutCanvas, stampsCanvas].forEach(c => {
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      c.style.width = img.width + 'px';
      c.style.height = img.height + 'px';
      c.style.left = img.offsetLeft + 'px';
      c.style.top = img.offsetTop + 'px';
    });
  }

  img.addEventListener('load', ()=>{
    resizeCanvases();
    clearCanvases();
  });
  window.addEventListener('resize', ()=>{ if (img.naturalWidth) resizeCanvases(); });

  function clearCanvases(){
    [overlaysCanvas, troutCanvas, stampsCanvas].forEach(c=>{
      const ctx = c.getContext('2d');
      ctx.clearRect(0,0,c.width,c.height);
    });
    currentPoints = [];
  }

  function buildOceanPixelList(threshold){
    oceanPixels = [];
    if (!img.naturalWidth) return oceanPixels;
    readCanvas.width = img.naturalWidth;
    readCanvas.height = img.naturalHeight;
    readCtx.clearRect(0,0,readCanvas.width, readCanvas.height);
    readCtx.drawImage(img, 0, 0, readCanvas.width, readCanvas.height);
    const data = readCtx.getImageData(0,0, readCanvas.width, readCanvas.height).data;
    for (let y=0;y<readCanvas.height;y++){
      const row = y*readCanvas.width*4;
      for (let x=0;x<readCanvas.width;x++){
        const i = row + x*4;
        const r = data[i], g = data[i+1], b = data[i+2], a = data[i+3];
        if (a === 0) continue;
        const lum = 0.2126*r + 0.7152*g + 0.0722*b;
        if (lum >= threshold) oceanPixels.push({x,y});
      }
    }
    return oceanPixels;
  }

  function sampleFlowAt(x,y, params){
    // sample noise at scaled coords and bias east
    const s = params.noiseScale;
    const nx = x * s / img.naturalWidth;
    const ny = y * s / img.naturalHeight;
    const fx = noise2(nx*10, ny*10) * 0.8; // noise in [-1,1]
    const fy = noise3(nx*10, ny*10) * 0.6;
    // base east vector
    const bx = 1;
    const by = 0;
    let vx = bx * params.flowStrength + fx * params.flowStrength;
    let vy = by * params.flowStrength + fy * params.flowStrength * 0.7;
    const len = Math.hypot(vx,vy) || 1;
    vx /= len; vy /= len;
    return {x: vx, y: vy};
  }

  function randomSample(arr, n, rng){
    if (n >= arr.length) return arr.slice(0);
    const res = new Array(n);
    for (let i=0;i<n;i++){
      const j = i + Math.floor(rng() * (arr.length - i));
      const tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
    for (let i=0;i<n;i++) res[i]=arr[i];
    return res;
  }

  // main generator
  function generatePoints(){
    if (!img.naturalWidth) { alert('Image not loaded'); return; }
    const n = Math.max(1, Math.min(400, parseInt(countEl.value || 0)));
    const threshold = parseInt(thresholdEl.value);
    const flowStrength = parseFloat(flowStrengthEl.value);
    const noiseScale = parseFloat(noiseScaleEl.value);
    const diffusion = parseFloat(diffusionEl.value);
    const stepSize = parseFloat(stepSizeEl.value);
    const maxSteps = parseInt(maxStepsEl.value);
    const seed = parseInt(seedEl.value) || 1234;

    // prepare noise with seed
    noise2 = makeNoise2D((seed*101 + 17) >>> 0);
    noise3 = makeNoise2D((seed*761 + 97) >>> 0);
    const rng = mulberry32(seed >>> 0);

    buildOceanPixelList(threshold);
    if (oceanPixels.length === 0){ alert('No ocean pixels at that threshold. Lower threshold.'); return; }

    currentPoints = [];
    const params = {flowStrength, noiseScale};

    // we'll try seeds from ocean pixels randomly and advect
    const attemptsLimit = n * 200; // safety
    let attempts = 0;
    let streakId = 0;

    while (currentPoints.length < n && attempts < attemptsLimit){
      attempts++;
      // pick random ocean seed
      const s = oceanPixels[Math.floor(rng() * oceanPixels.length)];
      let px = s.x + 0.001; // avoid exact ints
      let py = s.y + 0.001;
      streakId++;
      for (let step=0; step<maxSteps && currentPoints.length < n; step++){
        const f = sampleFlowAt(px, py, {flowStrength, noiseScale});
        // add diffusion jitter
        const jitterX = (rng() - 0.5) * 2 * diffusion;
        const jitterY = (rng() - 0.5) * 2 * diffusion;
        px += (f.x * stepSize) + jitterX;
        py += (f.y * stepSize) + jitterY;
        const ix = Math.round(px), iy = Math.round(py);
        if (ix < 0 || iy < 0 || ix >= img.naturalWidth || iy >= img.naturalHeight) break;
        // check ocean mask quickly by sampling pixel
        const idx = (iy * img.naturalWidth + ix) * 4;
        // get pixel from readCanvas
        const d = readCtx.getImageData(ix, iy, 1, 1).data;
        const r = d[0], g = d[1], b = d[2], a = d[3];
        if (a === 0) continue;
        const lum = 0.2126*r + 0.7152*g + 0.0722*b;
        if (lum >= threshold){
          currentPoints.push({x: ix, y: iy, streakId, step});
        }
      }
    }

    if (currentPoints.length === 0){ alert('No points generated; try changing parameters.'); return; }
    // if too many duplicates, dedupe by x,y
    const uniq = [];
    const seen = new Set();
    for (const p of currentPoints){
      const key = p.x + ',' + p.y;
      if (!seen.has(key)) { seen.add(key); uniq.push(p); }
      if (uniq.length >= n) break;
    }
    currentPoints = uniq.slice(0, n);
    drawPoints();
  }

  function drawPoints(){
    resizeCanvases();
    const ctx = troutCanvas.getContext('2d');
    ctx.clearRect(0,0,troutCanvas.width, troutCanvas.height);
    ctx.save();
    ctx.fillStyle = '#00ffd6';
    ctx.strokeStyle = 'rgba(0,0,0,0.6)';
    ctx.lineWidth = Math.max(1, troutCanvas.width / 1200);
    for (const p of currentPoints){
      ctx.beginPath();
      ctx.arc(p.x + 0.5, p.y + 0.5, 3, 0, Math.PI*2);
      ctx.fill(); ctx.stroke();
    }
    ctx.restore();
  }

  // exports
  function exportCSV(){
    if (!currentPoints.length) { alert('No points to export'); return; }
    const lines = ['x,y,streakId,step'];
    for (const p of currentPoints){ lines.push(`${p.x},${p.y},${p.streakId || 0},${p.step || 0}`); }
    const blob = new Blob([lines.join('\n')], {type:'text/csv'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'trout_points.csv'; a.click(); URL.revokeObjectURL(url);
  }
  function exportJSON(){
    if (!currentPoints.length) { alert('No points to export'); return; }
    const blob = new Blob([JSON.stringify(currentPoints, null, 2)], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'trout_points.json'; a.click(); URL.revokeObjectURL(url);
  }

  function exportPNG(){
    // composite: image -> overlays -> trout -> stamps
    // draw into an offscreen canvas at natural resolution
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0,0);
    ctx.drawImage(overlaysCanvas, 0,0);
    ctx.drawImage(troutCanvas, 0,0);
    ctx.drawImage(stampsCanvas, 0,0);
    canvas.toBlob(blob => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = 'trout_composite.png'; a.click(); URL.revokeObjectURL(url);
    });
  }

  // --- overlays minimal implementation ---
  // Upload shape and add layer with auto-stamp
  addOverlayBtn.addEventListener('click', async ()=>{
    const f = overlayUpload.files && overlayUpload.files[0];
    if (!f){ alert('Choose a PNG or SVG to upload first'); return; }
    const size = parseInt(stampSizeEl.value) || 32;
    const count = parseInt(stampCountEl.value) || 10;
    const opacity = parseFloat(stampOpacityEl.value) || 1;
    const randomRot = stampRotateEl.checked;
    const restrictOcean = restrictOceanEl.checked;
    const reader = new FileReader();
    reader.onload = async ()=>{
      const imgEl = new Image();
      imgEl.onload = ()=>{
        const layer = {id: Date.now(), name: f.name, img: imgEl, size, opacity, randomRot, restrictOcean, stamps: []};
        // auto-stamp randomly
        // build ocean list if needed
        if (restrictOcean && oceanPixels.length === 0) buildOceanPixelList(parseInt(thresholdEl.value));
        const rng = mulberry32((parseInt(seedEl.value) || 1234) >>> 0);
        for (let i=0;i<count;i++){
          let placed = false;
          for (let a=0;a<200 && !placed;a++){
            const x = Math.floor(rng() * img.naturalWidth);
            const y = Math.floor(rng() * img.naturalHeight);
            if (restrictOcean){
              const key = (y * img.naturalWidth + x) * 4;
              const d = readCtx.getImageData(x,y,1,1).data;
              const lum = 0.2126*d[0] + 0.7152*d[1] + 0.0722*d[2];
              if (lum < parseInt(thresholdEl.value)) continue; // skip non-ocean
            }
            const rot = randomRot ? (rng()*360) : 0;
            layer.stamps.push({x,y,rot});
            placed = true;
          }
        }
        overlays.push(layer);
        renderOverlays();
        refreshOverlayList();
      };
      imgEl.src = reader.result;
    };
    reader.readAsDataURL(f);
  });

  function renderOverlays(){
    const ctx = overlaysCanvas.getContext('2d');
    ctx.clearRect(0,0,overlaysCanvas.width, overlaysCanvas.height);
    for (const layer of overlays){
      ctx.save(); ctx.globalAlpha = layer.opacity || 1;
      for (const s of layer.stamps){
        ctx.translate(s.x, s.y);
        ctx.rotate((s.rot||0) * Math.PI/180);
        const half = layer.size/2;
        ctx.drawImage(layer.img, -half, -half, layer.size, layer.size);
        ctx.rotate(-(s.rot||0) * Math.PI/180);
        ctx.translate(-s.x, -s.y);
      }
      ctx.restore();
    }
  }

  function refreshOverlayList(){
    overlayList.innerHTML = '';
    overlays.forEach(layer=>{
      const el = document.createElement('div'); el.className='overlay-item';
      const imgThumb = document.createElement('img'); imgThumb.src = layer.img.src;
      const name = document.createElement('div'); name.textContent = layer.name; name.style.flex='1'; name.style.color='#cfe7ef';
      const btn = document.createElement('button'); btn.textContent='Remove'; btn.onclick = ()=>{ overlays = overlays.filter(l=>l !== layer); renderOverlays(); refreshOverlayList(); };
      el.appendChild(imgThumb); el.appendChild(name); el.appendChild(btn);
      overlayList.appendChild(el);
    });
  }

  // manual stamping by clicking on stampsCanvas
  let stampingLayer = null; // when uploading a single shape for manual stamping we can set this
  stampsCanvas.addEventListener('click', (ev)=>{
    if (!overlays.length) return; // nothing to stamp
    // default: stamp onto last layer
    const layer = overlays[overlays.length-1];
    const rect = stampsCanvas.getBoundingClientRect();
    const x = Math.round((ev.clientX - rect.left) * (stampsCanvas.width / rect.width));
    const y = Math.round((ev.clientY - rect.top) * (stampsCanvas.height / rect.height));
    const rot = layer.randomRot ? (Math.random()*360) : 0;
    layer.stamps.push({x,y,rot});
    renderOverlays(); refreshOverlayList();
  });

  // wire controls
  generateBtn.addEventListener('click', ()=>{
    // prepare read canvas with image pixels
    if (!img.naturalWidth){ alert('Image not loaded'); return; }
    readCanvas.width = img.naturalWidth; readCanvas.height = img.naturalHeight;
    readCtx.drawImage(img,0,0, readCanvas.width, readCanvas.height);
    buildOceanPixelList(parseInt(thresholdEl.value));
    generatePoints();
  });
  clearBtn.addEventListener('click', ()=>{ clearCanvases(); overlays = []; renderOverlays(); refreshOverlayList(); });
  exportCsv.addEventListener('click', exportCSV);
  exportJson.addEventListener('click', exportJSON);
  exportPng.addEventListener('click', exportPNG);

  // ensure canvases sized on start if image loaded
  if (img.complete && img.naturalWidth) { resizeCanvases(); }

})();
