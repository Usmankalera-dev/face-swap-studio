# FaceSwap Studio

A browser-only face swapping system for photos and video. No server, no uploads — everything runs on the visitor's device.

## Live site

Once GitHub Pages is enabled for this repo (Settings → Pages → Deploy from branch `main`, folder `/ (root)`), the app is live at:

https://usmankalera-dev.github.io/face-swap-studio/

## How it works

1. **Source face** — upload a clear, front-facing photo.
2. **Target** — upload a photo or a video whose face should be replaced.
3. **Swap** — press Swap.
   - Photo: the swapped image appears on the result canvas, downloadable as PNG.
   - Video: a live swapped preview plays on the result canvas. Press Record, play the clip, then Stop to download the recording (MP4 where supported, otherwise WebM).

Under the hood: face-api.js detects one face and 68 facial landmarks in the source and the target. The source face is warped onto the target with Delaunay triangles built from the source landmarks, colour-matched to the target in photo mode, and blended through a feathered face-hull mask. Video reuses the source triangulation for every frame.

## Quality notes (honest limits)

- Front-facing faces with a similar angle swap best. Side profiles, sunglasses and covered faces will look off — that is a limit of landmark-based swapping.
- The swap replaces the face region only. Hair, ears and background stay from the target.
- Video is processed live in the browser, so short clips on desktop Chrome work best.
- This is a landmark-warp swap, not a neural inswapper. It is fast, private and dependency-light, but it will not be pixel-perfect on every angle or lighting setup.

## Responsible use

Only swap faces you have permission to use — your own face, or someone who agreed. Do not use this to impersonate anyone, mislead people, or make deceptive videos.

## Run locally

Open `index.html` in a browser (an internet connection is needed the first time, to load face-api.js, Delaunator and the face models from CDN).

Built by Usman Kalera.
