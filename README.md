<p align="center">
  <img src="enegmatic/assets/enegmatic.png" width="112" alt="eNegmatic icon">
</p>

<h1 align="center">eNegmatic</h1>

<p align="center"><i>made by Rtx9 &amp; Claude</i></p>
So yea "another vibe coded ai slop app", this one differs in the way that i actually have experience with it and gonna have my local photo store test it out in a real enviroment.
This free, open-source "alternative to some others" desktop app converts scanned **colour and B&W film negatives** into positives. Tested on scans from a Nikon Z7
<img src="photos/example1.webp" width="48%" alt="Example 1"><img src="photos/example1-2.webp" width="48%" alt="Example 1">

<img src="photos/example2.webp" width="48%" alt="Example 2"><img src="photos/example2-2.webp" width="48%" alt="Example 2">

<img src="photos/example3.webp" width="48%" alt="Example 3"><img src="photos/example3-2.webp" width="48%" alt="Example 3">

<img src="photos/example4.webp" width="48%" alt="Example 4"><img src="photos/example4-2.webp" width="48%" alt="Example 4">


## Download (Windows)

Get the latest `eNegmatic-…-windows.zip` from the **[Releases](../../releases)** page, unzip it anywhere,
and run `eNegmatic.exe`. No installation or Python needed.

The app isn't code-signed, so the first launch may show *"Windows protected your PC"*.
Click **More info → Run anyway**.

## Run from source

Double-click **`run.bat`**. On first run it creates `.venv` and installs the dependencies.

Or run it manually (Python 3.10 or newer, any OS):

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py            # optionally: main.py <files or folders>
```

## Build the Windows app yourself

Double-click **`build.bat`**. It installs PyInstaller and writes `dist\eNegmatic\eNegmatic.exe`
plus `dist\eNegmatic-windows.zip`.

Releases are built automatically: pushing a version tag (for example `v1.0.0`) runs the
GitHub Actions workflow in `.github/workflows/build.yml`. It builds the app and attaches the zip to a new Release.

## Features

- **Automatic inversion.** The orange mask is removed and channels balanced per frame (auto levels + auto white balance).
- **Colour and B&W modes, detected automatically.** Each new frame is classified on load. B&W silver puts all three channels on one line in density space, while colour dyes spread out; the orange mask is a second check. Click either button to override. B&W is processed as a single channel and can be exported as true grayscale.
- **Auto crop.** Finds the picture edges inside the rebate, the gaps between frames and the holder, and straightens tilt up to ±4°.
  It looks for straight, frame-length edges with a uniform band outside them, so it works for colour and B&W, 35 mm and medium format, and night scenes.
  New images are auto-cropped when they load (switch this off under *Options*).
- **Manual crop tool.** Drag handles, rule-of-thirds overlay and aspect locks (3:2, 4:3, 5:4, 1:1, 6×7, 16:9).
  Also: 90° rotate, flip, and fine straightening.
- **Pickers.** *Film base* (click the clear film between frames for the most accurate colour) and *Pick neutral*
  (click something grey to set white balance).
- **Adjustments.** Exposure, contrast, midtones, highlights, shadows, black and white points,
  temperature, tint, saturation, sharpening. Double-click any slider to reset it.
- **Zoom.** Click the image to toggle fit and 1:1, or use mouse wheel or touchpad pinch to zoom; drag or two-finger scroll to pan.
  Past the preview's resolution, the full-resolution raw is decoded in the background and only the visible area is rendered sharply.
- **Live histogram** and a *Negative* toggle (**B**) for before/after.
- **Copy / Paste / Paste to all** settings between frames.
  The ⋯ button next to Copy chooses which groups get copied: film and inversion, tone, colour, detail, rotation, crop.
  *Lock analysis* freezes one frame's base and levels, so a whole roll converts consistently.
- **Export** the current, selected or all frames as JPEG, TIFF or PNG (8/16-bit).
  You can resize, add a suffix, and choose the destination; exports run in parallel and can be cancelled.
- **Collapsible panel sections**, each with its own reset button, remembered between sessions.
- **Dark / light / system theme** (**Ctrl+T**).
- **Edits are saved automatically** in a small `.enegmatic.json` next to your images and restored when you reopen them.

## Suggested workflow for a roll

1. *Open Folder…* with the roll's NEFs. Frames auto-crop as previews build.
2. Pick *Colour negative* or *B&W negative*.
3. Get one representative frame right. Pick the film base if the border is visible, then adjust.
4. Optional: tick **Lock analysis (roll)** for consistent colour across frames.
5. **Copy**, then **Paste to all**. Fix any odd frames, then **Export All…**.

## Shortcuts

| Key | Action |
|---|---|
| ← / → | Previous / next frame (click the image first) |
| Click / Z | Zoom 1:1 ↔ fit |
| Wheel, pinch, Ctrl+= / Ctrl+- | Zoom in / out |
| Ctrl+0 | Fit to window |
| C | Crop tool (Enter applies, Esc cancels) |
| A | Auto crop |
| B | Show the original negative |
| [ / ] | Rotate left / right |
| Shift+F | Flip |
| Ctrl+Shift+C / V | Copy / paste settings |
| Ctrl+Alt+C | Choose what to copy |
| Ctrl+R | Reset adjustments |
| Ctrl+E / Ctrl+Shift+E | Export current / export all |

## How it works

Raw files are decoded linearly with a fixed daylight white balance, so frames stay consistent regardless of camera
AWB. The negative is converted to **density** (−log10 of transmittance). Each channel is then normalised between its
film-base density and its densest point, which removes the orange mask. Optional grey-pixel balancing follows.
Density is mapped back to light with an exponential "print" curve, and the tone and colour adjustments are applied.
Every per-channel step is baked into one lookup table, which keeps the preview fast.

The preview and the export share the same stored analysis, so exported files match what you see.

## Files

- `main.py` is the entry point; it shows the splash screen while the app loads.
- `enegmatic/splash.py` draws the splash screen and app icon in Century Gothic (ships with Windows / Office), falling back to the bundled Comfortaa font (`enegmatic/assets`, SIL Open Font License).
- `enegmatic/core.py` holds the conversion pipeline (no Qt).
- `enegmatic/autocrop.py` does frame detection and straightening.
- `enegmatic/fileio.py` covers raw and bitmap loading, saving, and the sidecar files.
- `enegmatic/batch.py` runs full-resolution export in worker processes.
- `enegmatic/mainwindow.py`, `widgets.py`, `export.py` and `theme.py` make up the UI.
- `packaging/` holds the PyInstaller spec and the Windows file-version info; `build.bat` runs it.
- `tools/make_icon.py` regenerates `enegmatic/assets/enegmatic.ico` and the logo PNG.

## Notes

- Full-resolution exports use roughly 1–2 GB of RAM per parallel job for 45 MP raws.
  Lower *Parallel jobs* in the export dialog if memory is tight.
- Auto crop needs some visible border (film base, gap, holder or light). For pre-cropped scans it leaves the frame alone.
  It leaves a 1 % safety inset, which you can change under *Options ▸ Auto-crop Inset*.

## License

eNegmatic is free software, released under the **GNU General Public License v3.0** (see [LICENSE](LICENSE)).
You may use, study, share and modify it. If you distribute a modified version, it must also be released under the GPL v3.0, with its source code.

Bundled third-party components keep their own licences; see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
