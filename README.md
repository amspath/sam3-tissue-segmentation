# ai-assisted-wsi-labeling

Two CLI tools for whole-slide-image (WSI) tissue labeling:

- **`wsi-thumb`** — downsample a WSI to a viewable PNG thumbnail.
- **`sam3-segment`** — segment tissue from a text prompt with [SAM 3](sam3/),
  on either a plain image *or* a raw WSI (auto-thumbnailed). For WSIs it also
  exports full-resolution polygon coordinates.

## Install

Requires Python ≥ 3.13, [uv](https://docs.astral.sh/uv/), and a CUDA GPU for
`sam3-segment` (CPU works but is very slow).

```bash
# 1. Clone with the SAM 3 submodule
git clone --recurse-submodules <repo-url>
cd ai-assisted-wsi-labeling
# (or, in an existing clone: git submodule update --init)

# 2. Base env: fastslide, huggingface-hub, pillow
uv sync

# 3. SAM 3 (editable install of the submodule) + its runtime deps
uv pip install -e sam3
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install matplotlib opencv-python
```

Verified working versions in this venv: `torch 2.11.0+cu128`, `matplotlib
3.11.0`, `opencv-python 5.0.0`, `numpy 2.5.1`, Python 3.13.

The SAM 3 checkpoint (`facebook/sam3.1`) downloads automatically via `hf` on the
first `sam3-segment` run and is cached under `~/.cache/huggingface`.

> Note: `torch`/`sam3`/`matplotlib`/`opencv` are not in `pyproject.toml`
> dependencies — they need the CUDA index / editable submodule and are installed
> explicitly above.

## wsi-thumb

Make a downsampled thumbnail of a whole-slide image.

```bash
uv run wsi-thumb <slide> [-d FACTOR] [-o OUTPUT]
```

- `-d`, `--downsample` — downsample factor (default: 5).
- `-o`, `--output` — output PNG path (default: `<slide>_thumb<D>x.png`).

```bash
uv run wsi-thumb RT14-09099_HE.tiff -d 10
# (163328, 46592) -> (16333, 4659)  RT14-09099_HE_thumb10x.png
```

## sam3-segment

Segment tissue with SAM 3 from a text prompt. The input can be a normal image
**or** a raw WSI — WSIs are automatically read and downsampled to a 2048px
thumbnail (auto-computed downsample; SAM 3 detects internally at 1008px, so
2048 keeps masks crisp without wasted work).

```bash
uv run sam3-segment <input> [prompt] [-t THRESHOLD] [-o PREFIX]
```

- `input` — an image (PNG/JPG…) or a WSI (`.tiff`, `.isyntax`, `.svs`,
  `.ndpi`, …). Detected by file extension.
- `prompt` — text prompt (default: `histological tissue blob`).
- `-t`, `--threshold` — confidence threshold (default: 0.5). Only detections at
  or above this score are kept.
- `-o`, `--output-prefix` — output prefix (default: `<input>_<prompt>`).

### Outputs

- `<prefix>_overlay.png` — masks + boxes + scores drawn on the image.
- `<prefix>_mask.png` — binary mask, union of all detected instances (2048px
  longest side).
- `<prefix>_coords.json` — **WSI inputs only.** Polygon contours in
  **full-resolution (level-0) WSI pixel coordinates**, so they map back onto the
  original slide. Plain JSON:

  ```json
  [{"score": 0.82, "polygon": [[x, y], [x, y], ...]}, ...]
  ```

```bash
# WSI: auto-thumbnails, writes overlay + mask + level-0 coords
uv run sam3-segment RT14-09099_HE.tiff
# found 2 object(s) for prompt 'histological tissue blob': scores=[0.82, 0.8]
# wrote RT14-09099_HE_histological_tissue_blob_{overlay,mask}.png
# wrote RT14-09099_HE_histological_tissue_blob_coords.json (9 polygon(s), level-0 coordinates)

# Plain image: overlay + mask only
uv run sam3-segment RT14-09099_HE_thumb10x.png
```
