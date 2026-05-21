"""
WSI Viewer — FastAPI backend.

Tile protocol (Deep Zoom Image / DZI):
  GET /api/slides/{slide_id}.dzi                      → XML descriptor
  GET /api/slides/{slide_id}_files/{z}/{col}_{row}.jpeg → JPEG tile

Supporting endpoints:
  GET /api/browse?path=<dir>          → directory listing (with slide_id)
  GET /api/slides/{slide_id}/info     → JSON slide metadata
  GET /api/slides/{slide_id}/thumbnail → JPEG thumbnail ≤ 512 px
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
# Package-relative paths
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = PACKAGE_DIR / "static"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE_SIZE = 254     # DZI logical tile size (px)
TILE_OVERLAP = 1    # overlap on each edge → actual render is 256 × 256

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".svs", ".tiff", ".tif", ".qptiff", ".mrxs", ".ndpi", ".scn", ".czi",
    ".vms", ".vmu", ".bif",
})

# ---------------------------------------------------------------------------
# Process-wide shared resources
# ---------------------------------------------------------------------------

_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
_cache_manager = CacheManager.create(capacity_bytes=1 * 1024**3)  # 1 GB
_dzi_info_cache: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Slide ID  — URL-safe base64 encoding of the absolute file path
# ---------------------------------------------------------------------------


def encode_slide_id(path: str) -> str:
    """Encode an absolute file path as a URL-safe base64 slide ID."""
    return base64.urlsafe_b64encode(path.encode()).decode().rstrip("=")


def decode_slide_id(slide_id: str) -> str:
    """Decode a slide ID back to an absolute file path, raising HTTP errors on invalid input."""
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
# SlideManager — async LRU cache of open FastSlide file handles
# ---------------------------------------------------------------------------


class SlideManager:
    """Keeps the last *max_open* slides open to avoid repeated open/close overhead."""

    def __init__(self, max_open: int = 8) -> None:
        self._max_open = max_open
        self._slides: OrderedDict[str, FastSlide] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, path: str) -> FastSlide:
        async with self._lock:
            if path in self._slides:
                self._slides.move_to_end(path)
                return self._slides[path]

            # Evict the least-recently-used slide when at capacity
            while len(self._slides) >= self._max_open:
                _, evicted = self._slides.popitem(last=False)
                try:
                    evicted.close()
                except Exception:
                    pass

            slide: FastSlide = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: FastSlide.from_file_path(path),
            )
            slide.set_cache(_cache_manager)
            self._slides[path] = slide
            return slide

    async def close_all(self) -> None:
        async with self._lock:
            for slide in self._slides.values():
                try:
                    slide.close()
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
# Image normalisation — coerce any FastSlide output to uint8 RGB
# ---------------------------------------------------------------------------


def _normalize_to_rgb(arr: np.ndarray, dtype_str: str) -> np.ndarray:
    # Bit-depth
    if dtype_str == "uint16":
        arr = (arr.astype(np.float32) / 256.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # Channel count
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)          # grayscale → RGB
    elif arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)    # 1-ch → RGB
    elif arr.shape[2] == 4:
        # RGBA → composite over white background
        rgb = arr[:, :, :3].astype(np.float32)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        arr = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    # Already 3-channel RGB — pass through

    return arr


def _blank_jpeg(w: int, h: int) -> bytes:
    img = PILImage.new("RGB", (max(w, 1), max(h, 1)), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tile renderer — runs inside the thread-pool executor
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

    # 1. DZI level → downsample factor relative to full resolution
    dzi_downsample = 2 ** (max_dzi_level - dzi_level)

    # 2. Pick the closest-matching slide pyramid level
    slide_level = slide.get_best_level_for_downsample(float(dzi_downsample))
    actual_ds = slide.level_downsamples[slide_level]
    level_w, level_h = slide.level_dimensions[slide_level]

    # 3. Tile extent in level-0 pixels (including overlap margin)
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

    # 4. Convert origin to level-native coordinates (FastSlide convention)
    lx, ly = slide.convert_level0_to_level_native(x_start_l0, y_start_l0, level=slide_level)

    # 5. How many level-native pixels to request
    read_w = min(round(tile_w_l0 / actual_ds), max(0, level_w - lx))
    read_h = min(round(tile_h_l0 / actual_ds), max(0, level_h - ly))

    if read_w <= 0 or read_h <= 0:
        return _blank_jpeg(ts + 2 * ov, ts + 2 * ov)

    # 6. Read the region
    fs_img = slide.read_region(location=(lx, ly), level=slide_level, size=(read_w, read_h))
    arr = _normalize_to_rgb(np.asarray(fs_img.numpy()), fs_img.dtype)

    # 7. Resize to the intended output pixel size (edge tiles are smaller)
    out_w = max(math.ceil(tile_w_l0 / dzi_downsample), 1)
    out_h = max(math.ceil(tile_h_l0 / dzi_downsample), 1)

    pil_img = PILImage.fromarray(arr, "RGB")
    if pil_img.size != (out_w, out_h):
        pil_img = pil_img.resize((out_w, out_h), PILImage.LANCZOS)

    # 8. Encode — lower quality for overview zoom levels
    quality = 75 if dzi_level < max_dzi_level - 3 else 85
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


def _make_thumbnail(slide: FastSlide, max_size: int = 512) -> bytes:
    assoc = slide.associated_images
    if "thumbnail" in assoc:
        img = assoc["thumbnail"]
        arr = _normalize_to_rgb(np.asarray(img.numpy()), img.dtype)
    else:
        level = slide.level_count - 1
        lw, lh = slide.level_dimensions[level]
        img = slide.read_region((0, 0), level, (lw, lh))
        arr = _normalize_to_rgb(np.asarray(img.numpy()), img.dtype)

    pil = PILImage.fromarray(arr, "RGB")
    pil.thumbnail((max_size, max_size), PILImage.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


def create_app(default_browse_path: Path | None = None) -> FastAPI:
    """
    Create and return the FastAPI application.

    Parameters
    ----------
    default_browse_path:
        Directory the file-browser opens to on first load.
        Defaults to the current working directory.
    """
    browse_root = default_browse_path or Path.cwd()

    application = FastAPI(
        title="WSI Viewer",
        description="Web-based whole slide image viewer powered by FastSlide.",
        version="0.1.0",
    )

    application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @application.on_event("shutdown")
    async def _shutdown() -> None:
        await _slide_manager.close_all()

    # ------------------------------------------------------------------ #
    # Frontend
    # ------------------------------------------------------------------ #

    @application.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))

    # ------------------------------------------------------------------ #
    # File browser
    # ------------------------------------------------------------------ #

    @application.get("/api/browse", summary="List directory contents")
    async def browse(path: str = Query(default="")) -> JSONResponse:
        target = Path(path).resolve() if path else browse_root

        if not target.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a directory: {target}")

        try:
            items = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied")

        entries: list[dict[str, Any]] = []
        for p in items:
            if p.name.startswith("."):
                continue
            if p.is_dir():
                entries.append({"name": p.name, "path": str(p), "type": "dir"})
            elif p.is_file():
                name_lower = p.name.lower()
                if any(name_lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    entries.append({
                        "name": p.name,
                        "path": str(p),
                        "type": "file",
                        "size": p.stat().st_size,
                        "slide_id": encode_slide_id(str(p)),
                    })

        return JSONResponse({"path": str(target), "entries": entries})

    # ------------------------------------------------------------------ #
    # DZI descriptor
    # ------------------------------------------------------------------ #

    @application.get("/api/slides/{slide_id}.dzi", summary="DZI XML descriptor")
    async def get_dzi(slide_id: str) -> Response:
        path = decode_slide_id(slide_id)
        slide = await _slide_manager.get(path)
        info = await _get_dzi_info(path, slide)
        return Response(
            content=_make_dzi_xml(info),
            media_type="application/xml",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # ------------------------------------------------------------------ #
    # DZI tiles
    # ------------------------------------------------------------------ #

    @application.get(
        "/api/slides/{slide_id}_files/{dzi_level}/{tile_spec}.jpeg",
        summary="DZI tile",
    )
    async def get_tile(slide_id: str, dzi_level: int, tile_spec: str) -> Response:
        try:
            col_s, row_s = tile_spec.split("_")
            col, row = int(col_s), int(row_s)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid tile spec; expected {col}_{row}")

        path = decode_slide_id(slide_id)
        slide = await _slide_manager.get(path)
        info = await _get_dzi_info(path, slide)

        jpeg_bytes: bytes = await asyncio.get_event_loop().run_in_executor(
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

    # ------------------------------------------------------------------ #
    # Thumbnail
    # ------------------------------------------------------------------ #

    @application.get("/api/slides/{slide_id}/thumbnail", summary="Slide thumbnail (≤ 512 px)")
    async def get_thumbnail(slide_id: str) -> Response:
        path = decode_slide_id(slide_id)
        slide = await _slide_manager.get(path)
        jpeg_bytes: bytes = await asyncio.get_event_loop().run_in_executor(
            _executor,
            lambda: _make_thumbnail(slide),
        )
        return Response(
            content=jpeg_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # ------------------------------------------------------------------ #
    # Slide info
    # ------------------------------------------------------------------ #

    @application.get("/api/slides/{slide_id}/info", summary="Slide metadata")
    async def get_slide_info(slide_id: str) -> JSONResponse:
        path = decode_slide_id(slide_id)
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

    return application


# ---------------------------------------------------------------------------
# Module-level app instance (for `uvicorn wsi_labeling.server:app`)
# ---------------------------------------------------------------------------

app = create_app()
