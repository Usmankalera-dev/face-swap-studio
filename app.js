/* FaceSwap Studio — browser-only face swap.
   Method: face-api.js detects one face + 68 landmarks in source and target.
   The source face is warped onto the target with Delaunay triangles built
   from the source landmarks, colour-matched (photo mode), then blended
   through a feathered face-hull mask. Video uses the same warp live,
   reusing the source triangulation for every frame. */
(function () {
  'use strict';

  var MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights';
  var modelsReady = false;
  var mode = 'photo'; // 'photo' | 'video'

  var sourceImg = null, targetImg = null;
  var sourcePoints = null, sourceTriangles = null, sourceCanvas = null;
  var targetVideo = document.getElementById('targetVideo');
  var resultCanvas = document.getElementById('resultCanvas');
  var resultCtx = resultCanvas.getContext('2d');
  var latestTargetPoints = null, latestTargetVideoSize = null;
  var videoRunning = false, videoRaf = 0;
  var recorder = null, recordedChunks = [];

  function $(id) { return document.getElementById(id); }
  function setStatus(msg) { $('workStatus').textContent = msg; }
  function setModelStatus(msg) { $('modelStatus').textContent = msg; }

  function fileToImage(file) {
    return new Promise(function (resolve, reject) {
      var url = URL.createObjectURL(file);
      var img = new Image();
      img.onload = function () { URL.revokeObjectURL(url); resolve(img); };
      img.onerror = function () { reject(new Error('Could not read that image.')); };
      img.src = url;
    });
  }

  function updateSwapButton() {
    var ok = modelsReady && sourceImg && (mode === 'photo' ? targetImg : targetVideo.src);
    $('swapBtn').disabled = !ok;
  }

  // ---------- models ----------
  async function loadModels() {
    try {
      await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
      await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
      modelsReady = true;
      setModelStatus('Face models ready. Upload a source face and a target.');
      updateSwapButton();
    } catch (e) {
      setModelStatus('Could not load face models. Check your connection and reload the page.');
    }
  }

  async function detectPoints(input) {
    // First try a strict threshold, then retry looser so smaller / angled
    // faces are not missed entirely.
    var thresholds = [0.5, 0.3];
    for (var t = 0; t < thresholds.length; t++) {
      var det = await faceapi.detectSingleFace(
        input, new faceapi.TinyFaceDetectorOptions({ inputSize: 640, scoreThreshold: thresholds[t] })
      ).withFaceLandmarks();
      if (det) return det.landmarks.positions.map(function (p) { return [p.x, p.y]; });
    }
    return null;
  }

  // ---------- geometry ----------
  function convexHull(points) {
    var pts = points.slice().sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
    function cross(o, a, b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); }
    var lower = [];
    pts.forEach(function (p) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
      lower.push(p);
    });
    var upper = [];
    pts.slice().reverse().forEach(function (p) {
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
      upper.push(p);
    });
    lower.pop(); upper.pop();
    return lower.concat(upper);
  }

  function affineTransform(srcTri, dstTri) {
    var x1 = srcTri[0], y1 = srcTri[1], x2 = srcTri[2], y2 = srcTri[3], x3 = srcTri[4], y3 = srcTri[5];
    var u1 = dstTri[0], v1 = dstTri[1], u2 = dstTri[2], v2 = dstTri[3], u3 = dstTri[4], v3 = dstTri[5];
    var det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2);
    if (Math.abs(det) < 1e-6) return null;
    var a = (u1 * (y2 - y3) + u2 * (y3 - y1) + u3 * (y1 - y2)) / det;
    var b = (v1 * (y2 - y3) + v2 * (y3 - y1) + v3 * (y1 - y2)) / det;
    var c = (u1 * (x3 - x2) + u2 * (x1 - x3) + u3 * (x2 - x1)) / det;
    var d = (v1 * (x3 - x2) + v2 * (x1 - x3) + v3 * (x2 - x1)) / det;
    var e = (u1 * (x2 * y3 - x3 * y2) + u2 * (x3 * y1 - x1 * y3) + u3 * (x1 * y2 - x2 * y1)) / det;
    var f = (v1 * (x2 * y3 - x3 * y2) + v2 * (x3 * y1 - x1 * y3) + v3 * (x1 * y2 - x2 * y1)) / det;
    return [a, b, c, d, e, f];
  }

  function drawWarpedFace(ctx, srcCanvas, srcPts, dstPts, triangles) {
    for (var i = 0; i < triangles.length; i += 3) {
      var i1 = triangles[i], i2 = triangles[i + 1], i3 = triangles[i + 2];
      var s = [srcPts[i1][0], srcPts[i1][1], srcPts[i2][0], srcPts[i2][1], srcPts[i3][0], srcPts[i3][1]];
      var d = [dstPts[i1][0], dstPts[i1][1], dstPts[i2][0], dstPts[i2][1], dstPts[i3][0], dstPts[i3][1]];
      var m = affineTransform(s, d);
      if (!m) continue;
      // Expand the clip triangle ~2% around its centroid. Neighbouring
      // triangles then overlap by a pixel or two, which hides the straight
      // facet seams that otherwise show across cheeks and chin. The affine
      // transform itself still uses the exact triangle, so the image content
      // is not stretched by this expansion.
      var cx = (d[0] + d[2] + d[4]) / 3, cy = (d[1] + d[3] + d[5]) / 3;
      var dc = [];
      for (var k = 0; k < 3; k++) {
        dc.push(cx + (d[k * 2] - cx) * 1.02, cy + (d[k * 2 + 1] - cy) * 1.02);
      }
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(dc[0], dc[1]); ctx.lineTo(dc[2], dc[3]); ctx.lineTo(dc[4], dc[5]);
      ctx.closePath(); ctx.clip();
      ctx.setTransform(m[0], m[1], m[2], m[3], m[4], m[5]);
      ctx.drawImage(srcCanvas, 0, 0);
      ctx.restore();
    }
  }

  // Forehead points: 68-point landmarks stop at the brows, so a plain hull
  // ends in a hard band below the hairline. Extend the brow points upward
  // along the nose-to-brow axis so the swapped region reaches the hairline.
  function foreheadPoints(targetPts) {
    var box = bbox(targetPts);
    var faceHeight = Math.max(1, box[3] - box[1]);
    var browCentre = [
      (targetPts[19][0] + targetPts[24][0]) / 2,
      (targetPts[19][1] + targetPts[24][1]) / 2
    ];
    var noseTip = targetPts[30];
    var ax = browCentre[0] - noseTip[0], ay = browCentre[1] - noseTip[1];
    var len = Math.sqrt(ax * ax + ay * ay) || 1;
    ax /= len; ay /= len;
    var out = [];
    for (var i = 17; i <= 26; i++) {
      out.push([targetPts[i][0] + ax * faceHeight * 0.38, targetPts[i][1] + ay * faceHeight * 0.38]);
    }
    return out;
  }

  function featheredMask(w, h, hull, blur) {
    // NOTE: the mask canvas must start TRANSPARENT, not black. A black
    // fill is fully opaque, so 'destination-in' would keep everything and
    // the feathered edge would never happen. White hull on transparent
    // background, blurred, gives a real alpha feather.
    var c = document.createElement('canvas'); c.width = w; c.height = h;
    var x = c.getContext('2d');
    x.fillStyle = '#fff';
    x.beginPath();
    hull.forEach(function (p, i) { if (i === 0) x.moveTo(p[0], p[1]); else x.lineTo(p[0], p[1]); });
    x.closePath(); x.fill();
    var blurred = document.createElement('canvas'); blurred.width = w; blurred.height = h;
    var bx = blurred.getContext('2d');
    bx.filter = 'blur(' + blur + 'px)';
    bx.drawImage(c, 0, 0);
    return blurred;
  }

  // Returns a canvas holding the target with the source face blended in.
  // softBlur: true for photos (hides seams), false for video (blur filters
  // per frame are a major cause of dropped frames and flicker).
  // maskBlur: feather radius; video uses a smaller radius for the same reason.
  function swapOnto(targetCanvas, targetPts, srcCanvas, srcPts, triangles, softBlur, maskBlur) {
    var w = targetCanvas.width, h = targetCanvas.height;
    var warped = document.createElement('canvas'); warped.width = w; warped.height = h;
    var wx = warped.getContext('2d');
    drawWarpedFace(wx, srcCanvas, srcPts, targetPts, triangles);
    var soft = warped;
    if (softBlur) {
      // Slight blur on the warped face softens any remaining triangle seams
      // before masking. If the browser ignores ctx.filter, the unblurred
      // warped face is still used, so this never breaks the swap.
      soft = document.createElement('canvas'); soft.width = w; soft.height = h;
      var sx0 = soft.getContext('2d');
      try { sx0.filter = 'blur(1.5px)'; } catch (e) {}
      sx0.drawImage(warped, 0, 0);
      try { sx0.filter = 'none'; } catch (e) {}
    }
    var hullPts = targetPts.concat(foreheadPoints(targetPts));
    var mask = featheredMask(w, h, convexHull(hullPts), maskBlur || Math.max(10, w / 60));
    var sx = soft.getContext('2d');
    sx.globalCompositeOperation = 'destination-in';
    sx.drawImage(mask, 0, 0);
    var out = document.createElement('canvas'); out.width = w; out.height = h;
    var ox = out.getContext('2d');
    ox.drawImage(targetCanvas, 0, 0);
    ox.drawImage(soft, 0, 0);
    return out;
  }

  function bbox(points) {
    var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    points.forEach(function (p) {
      minX = Math.min(minX, p[0]); minY = Math.min(minY, p[1]);
      maxX = Math.max(maxX, p[0]); maxY = Math.max(maxY, p[1]);
    });
    return [minX, minY, maxX, maxY];
  }

  // Mean colour sampled ONLY at landmark points (cheeks, forehead, chin),
  // never over the whole bounding box. The old box mean included the
  // background (e.g. a white studio backdrop), which shifted the face
  // colour far too dark or bright and made swaps look wrong.
  function meanColour(canvas, points) {
    var ctx = canvas.getContext('2d');
    var r = 0, g = 0, b = 0, n = 0;
    points.forEach(function (p) {
      var x = Math.max(0, Math.min(canvas.width - 1, Math.round(p[0])));
      var y = Math.max(0, Math.min(canvas.height - 1, Math.round(p[1])));
      var d = ctx.getImageData(x, y, 1, 1).data;
      r += d[0]; g += d[1]; b += d[2]; n++;
    });
    return n ? [r / n, g / n, b / n] : [128, 128, 128];
  }

  function clampShift(v) { return Math.max(-50, Math.min(50, v)); }

  // Shift the source face colours toward the target face colours (photo mode).
  function colourMatchedSource(srcCanvas, srcPts, targetCanvas, targetPts) {
    var sm = meanColour(srcCanvas, srcPts);
    var tm = meanColour(targetCanvas, targetPts);
    var dr = clampShift(tm[0] - sm[0]), dg = clampShift(tm[1] - sm[1]), db = clampShift(tm[2] - sm[2]);
    var c = document.createElement('canvas'); c.width = srcCanvas.width; c.height = srcCanvas.height;
    var ctx = c.getContext('2d');
    ctx.drawImage(srcCanvas, 0, 0);
    var box = bbox(srcPts);
    var x = Math.max(0, Math.floor(box[0])), y = Math.max(0, Math.floor(box[1]));
    var w = Math.min(c.width - x, Math.ceil(box[2] - box[0]));
    var h = Math.min(c.height - y, Math.ceil(box[3] - box[1]));
    var img = ctx.getImageData(x, y, w, h), data = img.data;
    for (var i = 0; i < data.length; i += 4) {
      data[i] = Math.max(0, Math.min(255, data[i] + dr));
      data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + dg));
      data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + db));
    }
    ctx.putImageData(img, x, y);
    return c;
  }

  function canvasFromImage(img) {
    var c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight;
    c.getContext('2d').drawImage(img, 0, 0);
    return c;
  }

  function showDownload(canvas, filename, label) {
    var link = $('downloadLink');
    link.href = canvas.toDataURL('image/png');
    link.download = filename;
    link.textContent = label;
    link.classList.remove('hidden');
  }

  // ---------- photo swap ----------
  async function swapPhoto() {
    setStatus('Detecting faces…');
    try {
      var srcPts = await detectPoints(sourceImg);
      if (!srcPts) { setStatus('No face found in the source photo. Use a clear, front-facing photo.'); return; }
      var tgtPts = await detectPoints(targetImg);
      if (!tgtPts) { setStatus('No face found in the target photo. Use a photo where the face is clearly visible.'); return; }
      var srcCanvas = canvasFromImage(sourceImg);
      var tgtCanvas = canvasFromImage(targetImg);
      var matched = colourMatchedSource(srcCanvas, srcPts, tgtCanvas, tgtPts);
      var tri = Delaunator.from(srcPts).triangles;
      var out = swapOnto(tgtCanvas, tgtPts, matched, srcPts, tri, true);
      resultCanvas.width = out.width; resultCanvas.height = out.height;
      resultCtx.drawImage(out, 0, 0);
      showDownload(resultCanvas, 'faceswap-photo.png', 'Download swapped photo (PNG)');
      setStatus('Done. If the blend looks off, try photos with a more similar angle and lighting.');
    } catch (e) {
      setStatus('Swap failed. Reload the page and try again with clearer photos.');
    }
  }

  // ---------- video swap (live preview + record) ----------
  function scalePoints(points, sx, sy) {
    return points.map(function (p) { return [p[0] * sx, p[1] * sy]; });
  }

  async function prepareVideoSource() {
    var srcPts = await detectPoints(sourceImg);
    if (!srcPts) { setStatus('No face found in the source photo. Use a clear, front-facing photo.'); return false; }
    sourcePoints = srcPts;
    sourceTriangles = Delaunator.from(srcPts).triangles;
    sourceCanvas = canvasFromImage(sourceImg);
    return true;
  }

  // Smoothed points actually drawn on screen. Raw detections jump frame
  // to frame (landmark jitter + slow async detection), which is exactly
  // what reads as flicker. Displayed points ease toward each new detection
  // instead of snapping to it.
  var displayedTargetPoints = null;

  function pointsCentre(points) {
    var sx = 0, sy = 0;
    points.forEach(function (p) { sx += p[0]; sy += p[1]; });
    return [sx / points.length, sy / points.length];
  }

  async function videoDetectLoop() {
    while (videoRunning) {
      try {
        var vw = targetVideo.videoWidth, vh = targetVideo.videoHeight;
        if (vw && !targetVideo.paused) {
          var pts = await detectPoints(targetVideo);
          if (pts) {
            var scaled = scalePoints(pts, resultCanvas.width / vw, resultCanvas.height / vh);
            // Outlier rejection: if the detected face centre jumped far from
            // the previous centre, this detection is almost certainly wrong
            // (another face, a false box, a blurred frame). Ignore it rather
            // than letting the swapped face teleport and flicker.
            if (latestTargetPoints) {
              var cOld = pointsCentre(latestTargetPoints);
              var cNew = pointsCentre(scaled);
              var box = bbox(latestTargetPoints);
              var faceW = Math.max(1, box[2] - box[0]);
              var jump = Math.sqrt((cNew[0] - cOld[0]) * (cNew[0] - cOld[0]) + (cNew[1] - cOld[1]) * (cNew[1] - cOld[1]));
              if (jump > faceW * 0.35) {
                scaled = null;
              }
            }
            if (scaled) {
              latestTargetPoints = scaled;
              latestTargetVideoSize = [vw, vh];
            }
          }
        }
      } catch (e) { /* keep last good landmarks */ }
      await new Promise(function (r) { setTimeout(r, 120); });
    }
  }

  function videoRenderLoop() {
    if (!videoRunning) return;
    var vw = targetVideo.videoWidth, vh = targetVideo.videoHeight;
    if (vw && !targetVideo.paused && !targetVideo.ended) {
      resultCtx.drawImage(targetVideo, 0, 0, resultCanvas.width, resultCanvas.height);
      if (latestTargetPoints && sourcePoints && sourceTriangles) {
        if (!displayedTargetPoints) {
          displayedTargetPoints = latestTargetPoints.map(function (p) { return [p[0], p[1]]; });
        } else {
          // Ease displayed points 40% toward the latest detection each
          // frame. This interpolates between slow detections and kills
          // the snap-to-new-landmarks flicker.
          for (var i = 0; i < displayedTargetPoints.length; i++) {
            displayedTargetPoints[i][0] += (latestTargetPoints[i][0] - displayedTargetPoints[i][0]) * 0.4;
            displayedTargetPoints[i][1] += (latestTargetPoints[i][1] - displayedTargetPoints[i][1]) * 0.4;
          }
        }
        // Note: source points stay in source-image coordinates; the affine
        // warp maps them onto the result-size target points. Do not scale them.
        var tmp = document.createElement('canvas');
        tmp.width = resultCanvas.width; tmp.height = resultCanvas.height;
        var tx = tmp.getContext('2d');
        tx.drawImage(targetVideo, 0, 0, resultCanvas.width, resultCanvas.height);
        // Video: no seam blur and a smaller mask feather, so frames keep up.
        var swapped = swapOnto(tmp, displayedTargetPoints, sourceCanvas, sourcePoints, sourceTriangles, false, 6);
        resultCtx.drawImage(swapped, 0, 0);
      }
    }
    videoRaf = requestAnimationFrame(videoRenderLoop);
  }

  async function swapVideo() {
    setStatus('Detecting the source face…');
    if (!await prepareVideoSource()) return;
    var vw = targetVideo.videoWidth, vh = targetVideo.videoHeight;
    if (!vw) { setStatus('Play the target video once first so its size is known, then press Swap again.'); return; }
    var scale = Math.min(1, 640 / vw);
    resultCanvas.width = Math.round(vw * scale);
    resultCanvas.height = Math.round(vh * scale);
    resultCtx.fillStyle = '#000'; resultCtx.fillRect(0, 0, resultCanvas.width, resultCanvas.height);
    displayedTargetPoints = null;
    latestTargetPoints = null;
    videoRunning = true;
    targetVideo.muted = true;
    targetVideo.currentTime = 0;
    await targetVideo.play().catch(function () {});
    videoDetectLoop();
    videoRenderLoop();
    $('recordBtn').classList.remove('hidden');
    $('downloadLink').classList.add('hidden');
    setStatus('Live swapped preview running. Press Record to save it, then Stop when the clip ends.');
  }

  function startRecording() {
    recordedChunks = [];
    var stream = resultCanvas.captureStream(30);
    var mime = '';
    ['video/mp4', 'video/webm;codecs=vp9', 'video/webm'].forEach(function (m) {
      if (!mime && window.MediaRecorder && MediaRecorder.isTypeSupported(m)) mime = m;
    });
    try {
      recorder = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 5000000 } : undefined);
    } catch (e) {
      setStatus('Recording is not supported in this browser. Try desktop Chrome.');
      return;
    }
    recorder.ondataavailable = function (e) { if (e.data.size) recordedChunks.push(e.data); };
    recorder.onstop = function () {
      var type = mime || 'video/webm';
      var blob = new Blob(recordedChunks, { type: type });
      var link = $('downloadLink');
      link.href = URL.createObjectURL(blob);
      link.download = type.indexOf('mp4') >= 0 ? 'faceswap-video.mp4' : 'faceswap-video.webm';
      link.textContent = 'Download recorded video';
      link.classList.remove('hidden');
      setStatus('Recording saved. Press Download to keep it.');
    };
    recorder.start();
    $('recordBtn').classList.add('hidden');
    $('stopBtn').classList.remove('hidden');
    setStatus('Recording… play the target video to capture the swapped version.');
    if (targetVideo.paused) targetVideo.play().catch(function () {});
  }

  function stopRecording() {
    if (recorder && recorder.state !== 'inactive') recorder.stop();
    $('stopBtn').classList.add('hidden');
    $('recordBtn').classList.remove('hidden');
  }

  // ---------- UI wiring ----------
  $('sourceInput').addEventListener('change', function (e) {
    var file = e.target.files[0]; if (!file) return;
    // Fresh object URL for the preview. The old code reused the Image
    // object's URL after fileToImage revoked it, so previews rendered as
    // thin broken strips instead of the photo.
    $('sourcePreview').src = URL.createObjectURL(file);
    fileToImage(file).then(function (img) {
      sourceImg = img;
      updateSwapButton();
    });
  });

  $('targetPhotoInput').addEventListener('change', function (e) {
    var file = e.target.files[0]; if (!file) return;
    $('targetPreview').src = URL.createObjectURL(file);
    fileToImage(file).then(function (img) {
      targetImg = img;
      updateSwapButton();
    });
  });

  $('targetVideoInput').addEventListener('change', function (e) {
    var file = e.target.files[0]; if (!file) return;
    targetVideo.src = URL.createObjectURL(file);
    targetVideo.load();
    updateSwapButton();
  });

  function setMode(m) {
    mode = m;
    $('tabPhoto').classList.toggle('active', m === 'photo');
    $('tabVideo').classList.toggle('active', m === 'video');
    $('targetPhotoBox').classList.toggle('hidden', m !== 'photo');
    $('targetVideoBox').classList.toggle('hidden', m !== 'video');
    $('recordBtn').classList.add('hidden');
    $('stopBtn').classList.add('hidden');
    videoRunning = false;
    cancelAnimationFrame(videoRaf);
    updateSwapButton();
  }
  $('tabPhoto').addEventListener('click', function () { setMode('photo'); });
  $('tabVideo').addEventListener('click', function () { setMode('video'); });

  $('swapBtn').addEventListener('click', function () {
    if (mode === 'photo') swapPhoto(); else swapVideo();
  });
  $('recordBtn').addEventListener('click', startRecording);
  $('stopBtn').addEventListener('click', stopRecording);

  loadModels();
})();
