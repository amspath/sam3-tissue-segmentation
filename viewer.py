"""
WSI Viewer backend — FastAPI + FastSlide + Deep Zoom Image (DZI) protocol.

Tile URL pattern (DZI):
  GET /api/slides/{slide_id}.dzi          → DZI XML descriptor
  GET /api/slides/{slide_id}_files/{z}/{col}_{row}.jpeg  → tile JPEG

File browser:
  GET /api/browse?path=<dir>              → JSON directory listing

Thumbnail:
  GET /api/slides/{slide_id}/thumbnail    → JPEG thumbnail ≤512px

Slide info:
  GET /api/slides/{slide_id}/info         → JSON metadata
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage

from fastslide import CacheManager, FastSlide

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE_SIZE = 254     # DZI logical tile size
TILE_OVERLAP = 1    # 1px overlap on each side → actual tiles are 256×256

SUPPORTED_EXTENSIONS = frozenset({
    ".svs", ".tiff", ".tif", ".qptiff", ".mrxs", ".ndpi", ".scn", ".czi",
    ".vms", ".vmu", ".bif",
})

PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_BROWSE_PATH = PROJECT_ROOT / "assets"

# ---------------------------------------------------------------------------
# Global resources
# ---------------------------------------------------------------------------

_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
_cache_manager = CacheManager.create(capacity_bytes=1 * 1024**3)  # 1 GB
_dzi_info_cache: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Slide ID encoding (URL-safe base64 of the absolute path)
# ---------------------------------------------------------------------------

def _encode_slide_id(path: str) -> str:
    return base64.urlsafe_b64encode(path.encode()).decode().rstrip("=")


def _decode_slide_id(slide_id: str) -> str:
    # Re-add padding
    remainder = len(slide_id) % 4
    if remainder:
        slide_id += "=" * (4 - remainder)
    try:
        path = base64.urlsafe_b64decode(slide_id.encode()).decode()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid slide ID") from exc
    p = Path(path).resolve()
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"Slide not found: {path}")
    return str(p)


# ---------------------------------------------------------------------------
# SlideManager — LRU cache of open FastSlide handles
# ---------------------------------------------------------------------------

class SlideManager:
    def __init__(self, max_open: int = 8) -> None:
        self._max_open = max_open
        self._slides: OrderedDict[str, FastSlide] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, path: str) -> FastSlide:
        async with self._lock:
            if path in self._slides:
                self._slides.move_to_end(path)
                return self._slides[path]

            # Evict least-recently-used if at capacity
            while len(self._slides) >= self._max_open:
                _, evicted = self._slides.popitem(last=False)
                try:
                    evicted.close()
                except Exception:
                    pass

            loop = asyncio.get_event_loop()
            slide: FastSlide = await loop.run_in_executor(
                _executor,
                lambda: FastSlide.from_file_path(path),
            )
            slide.set_cache(_cache_manager)
            self._slides[path] = slide
            return slide

    async def close_all(self) -> None:
        async with self._lock:
            for s in self._slides.values():
                try:
                    s.close()
                except Exception:
                    pass
            self._slides.clear()


_slide_manager = SlideManager()


# ---------------------------------------------------------------------------
# DZI helpers
# ---------------------------------------------------------------------------

def _compute_dzi_info(slide: FastSlide) -> dict[str, Any]:
    W, H = slide.dimensions
    max_dzi_level = math.ceil(math.log2(max(W, H))) if max(W, H) > 1 else 1
    return {
        "width": W,
        "height": H,
        "max_dzi_level": max_dzi_level,
        "tile_size": TILE_SIZE,
        "overlap": TILE_OVERLAP,
    }


async def _get_dzi_info(path: str, slide: FastSlide) -> dict[str, Any]:
    if path not in _dzi_info_cache:
        _dzi_info_cache[path] = _compute_dzi_info(slide)
    return _dzi_info_cache[path]


def _make_dzi_xml(info: dict[str, Any]) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"'
        f' Format="jpeg"'
        f' Overlap="{info["overlap"]}"'
        f' TileSize="{info["tile_size"]}">'
        f'<Size Width="{info["width"]}" Height="{info["height"]}"/>'
        "</Image>"
    )


# ---------------------------------------------------------------------------
# Image normalisation — always output uint8 RGB
# ---------------------------------------------------------------------------

def _normalize_to_rgb(arr: np.ndarray, dtype_str: str) -> np.ndarray:
    # Depth normalisation
    if dtype_str == "uint16":
        arr = (arr.astype(np.float32) / 256.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # Channel normalisation
    if arr.ndim == 2:
        # Grayscale HxW → HxWx3
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 4:
        # RGBA → RGB composited over white
        rgb = arr[:, :, :3].astype(np.float32)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        arr = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    # 3-channel already → pass through

    return arr


def _blank_jpeg(w: int, h: int) -> bytes:
    img = PILImage.new("RGB", (max(w, 1), max(h, 1)), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Core tile renderer (runs in thread pool)
# ---------------------------------------------------------------------------

def _render_tile(
    slide: FastSlide,
    dzi_level: int,
    col: int,
    row: int,
    info: dict[str, Any],
) -> bytes:
    W = info["width"]
    H = info["height"]
    max_dzi_level = info["max_dzi_level"]
    ts = info["tile_size"]   # 254
    ov = info["overlap"]     # 1

    # ---- Step 1: DZI → downsample factor ----
    dzi_downsample = 2 ** (max_dzi_level - dzi_level)

    # ---- Step 2: Best slide pyramid level ----
    slide_level = slide.get_best_level_for_downsample(float(dzi_downsample))
    actual_ds = slide.level_downsamples[slide_level]
    level_w, level_h = slide.level_dimensions[slide_level]

    # ---- Step 3: Tile extent in level-0 coordinates (with overlap) ----
    x0_l0 = col * ts * dzi_downsample
    y0_l0 = row * ts * dzi_downsample

    x_start_l0 = max(0, x0_l0 - ov * dzi_downsample)
    y_start_l0 = max(0, y0_l0 - ov * dzi_downsample)
    x_end_l0 = min(W, x0_l0 + (ts + ov) * dzi_downsample)
    y_end_l0 = min(H, y0_l0 + (ts + ov) * dzi_downsample)

    tile_w_l0 = x_end_l0 - x_start_l0
    tile_h_l0 = y_end_l0 - y_start_l0

    if tile_w_l0 <= 0 or tile_h_l0 <= 0:
        return _blank_jpeg(ts + 2 * ov, ts + 2 * ov)

    # ---- Step 4: Convert to level-native coordinates ----
    lx, ly = slide.convert_level0_to_level_native(x_start_l0, y_start_l0, level=slide_level)

    # ---- Step 5: How many level-native pixels to read ----
    read_w = round(tile_w_l0 / actual_ds)
    read_h = round(tile_h_l0 / actual_ds)

    # Clamp to level boundaries
    read_w = min(read_w, max(0, level_w - lx))
    read_h = min(read_h, max(0, level_h - ly))

    if read_w <= 0 or read_h <= 0:
        return _blank_jpeg(ts + 2 * ov, ts + 2 * ov)

    # ---- Step 6: Read from FastSlide ----
    fs_img = slide.read_region(location=(lx, ly), level=slide_level, size=(read_w, read_h))
    arr = np.asarray(fs_img.numpy())
    arr = _normalize_to_rgb(arr, fs_img.dtype)

    # ---- Step 7: Target output size (smaller at image edges) ----
    out_w = math.ceil(tile_w_l0 / dzi_downsample)
    out_h = math.ceil(tile_h_l0 / dzi_downsample)
    out_w = max(out_w, 1)
    out_h = max(out_h, 1)

    pil_img = PILImage.fromarray(arr, "RGB")
    if pil_img.size != (out_w, out_h):
        pil_img = pil_img.resize((out_w, out_h), PILImage.LANCZOS)

    # ---- Step 8: Encode as JPEG ----
    # Adaptive quality: lower for overview levels, higher for detail levels
    quality = 75 if dzi_level < max_dzi_level - 3 else 85

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


def _make_thumbnail(slide: FastSlide, max_size: int = 512) -> bytes:
    assoc = slide.associated_images
    if "thumbnail" in assoc:
        arr = np.asarray(assoc["thumbnail"].numpy())
        dtype_str = assoc["thumbnail"].dtype
    else:
        # Fall back to lowest-res pyramid level
        level = slide.level_count - 1
        lw, lh = slide.level_dimensions[level]
        img = slide.read_region((0, 0), level, (lw, lh))
        arr = np.asarray(img.numpy())
        dtype_str = img.dtype

    arr = _normalize_to_rgb(arr, dtype_str)
    pil = PILImage.fromarray(arr, "RGB")
    pil.thumbnail((max_size, max_size), PILImage.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="WSI Viewer", description="Web-based whole slide image viewer")

# Serve static files (index.html lives here)
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")


@app.on_event("shutdown")
async def _shutdown() -> None:
    await _slide_manager.close_all()


# ---- Frontend ----

@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(str(PROJECT_ROOT / "static" / "index.html"))


# ---- File browser ----

@app.get("/api/browse")
async def browse(path: str = Query(default="")) -> JSONResponse:
    if not path:
        target = DEFAULT_BROWSE_PATH if DEFAULT_BROWSE_PATH.is_dir() else Path.home()
    else:
        target = Path(path).resolve()

    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {target}")

    entries: list[dict[str, Any]] = []
    try:
        items = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    for p in items:
        if p.name.startswith("."):
            continue
        if p.is_dir():
            entries.append({"name": p.name, "path": str(p), "type": "dir"})
        elif p.is_file():
            # Check supported extension (handle double ext like .ome.tiff)
            name_lower = p.name.lower()
            if any(name_lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                entries.append({
                    "name": p.name,
                    "path": str(p),
                    "type": "file",
                    "size": p.stat().st_size,
                    "slide_id": _encode_slide_id(str(p)),
                })

    return JSONResponse({"path": str(target), "entries": entries})


# ---- DZI descriptor ----

@app.get("/api/slides/{slide_id}.dzi")
async def get_dzi(slide_id: str) -> Response:
    path = _decode_slide_id(slide_id)
    slide = await _slide_manager.get(path)
    info = await _get_dzi_info(path, slide)
    xml = _make_dzi_xml(info)
    return Response(
        content=xml,
        media_type="application/xml",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ---- DZI tiles ----

@app.get("/api/slides/{slide_id}_files/{dzi_level}/{tile_spec}.jpeg")
async def get_tile(slide_id: str, dzi_level: int, tile_spec: str) -> Response:
    try:
        col_s, row_s = tile_spec.split("_")
        col, row = int(col_s), int(row_s)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid tile spec")

    path = _decode_slide_id(slide_id)
    slide = await _slide_manager.get(path)
    info = await _get_dzi_info(path, slide)

    loop = asyncio.get_event_loop()
    jpeg_bytes: bytes = await loop.run_in_executor(
        _executor,
        lambda: _render_tile(slide, dzi_level, col, row, info),
    )

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Content-Length": str(len(jpeg_bytes)),
        },
    )


# ---- Thumbnail ----

@app.get("/api/slides/{slide_id}/thumbnail")
async def get_thumbnail(slide_id: str) -> Response:
    path = _decode_slide_id(slide_id)
    slide = await _slide_manager.get(path)
    loop = asyncio.get_event_loop()
    jpeg_bytes: bytes = await loop.run_in_executor(
        _executor,
        lambda: _make_thumbnail(slide),
    )
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ---- Slide info ----

@app.get("/api/slides/{slide_id}/info")
async def get_slide_info(slide_id: str) -> JSONResponse:
    path = _decode_slide_id(slide_id)
    slide = await _slide_manager.get(path)
    W, H = slide.dimensions
    mpp = slide.mpp
    return JSONResponse({
        "path": path,
        "width": W,
        "height": H,
        "level_count": slide.level_count,
        "level_dimensions": list(slide.level_dimensions),
        "level_downsamples": list(slide.level_downsamples),
        "format": slide.format,
        "dtype": slide.dtype,
        "mpp_x": mpp[0] if mpp else None,
        "mpp_y": mpp[1] if mpp else None,
    })
