import argparse
import contextlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import wsi_thumb
from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

DEFAULT_PROMPT = "histological tissue blobs"

# ponytail: extension heuristic; tiff assumed WSI in this project.
WSI_EXTS = {".isyntax", ".svs", ".ndpi", ".mrxs", ".scn", ".bif", ".vms", ".vmu", ".tif", ".tiff", ".svi"}


def load_input(path):
    """Return (RGB image <=2048px, scale) where scale=(sx, sy) maps image px -> level-0 WSI px, or None for a plain image."""
    if Path(path).suffix.lower() in WSI_EXTS:
        img, (w, h) = wsi_thumb.thumbnail(path)  # auto-targets 2048px longest side
        return img, (w / img.width, h / img.height)

    img = Image.open(path).convert("RGB")
    if max(img.size) > wsi_thumb.TARGET_PX:
        r = wsi_thumb.TARGET_PX / max(img.size)
        img = img.resize((round(img.width * r), round(img.height * r)), Image.LANCZOS)
    return img, None


def main():
    p = argparse.ArgumentParser(description="Segment an image or WSI with SAM3 from a text prompt.")
    p.add_argument("input", help="Path to an image or a whole-slide image (WSIs are auto-thumbnailed to 2048px)")
    p.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT,
                   help=f"Text prompt (default: {DEFAULT_PROMPT!r})")
    p.add_argument("-t", "--threshold", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    p.add_argument("-o", "--output-prefix", help="Output prefix (default: <input>_<prompt>)")
    args = p.parse_args()

    image, scale = load_input(args.input)

    # Use the already-downloaded sam3.1 checkpoint (cached, no re-download).
    ckpt = download_ckpt_from_hf(version="sam3.1")
    model = build_sam3_image_model(checkpoint_path=ckpt)
    processor = Sam3Processor(model, confidence_threshold=args.threshold)

    # SAM3's backbone expects a bf16 autocast context on CUDA (see the example
    # notebook, which enters it globally). CPU falls back to plain float32.
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with autocast:
        state = processor.set_image(image)
        state = processor.set_text_prompt(state=state, prompt=args.prompt)

    n = len(state["scores"])
    print(f"found {n} object(s) for prompt {args.prompt!r}: "
          f"scores={[round(s.item(), 2) for s in state['scores']]}")

    prefix = args.output_prefix or f"{args.input.rsplit('.', 1)[0]}_{args.prompt.replace(' ', '_')}"

    plot_results(image, state)
    plt.axis("off")
    plt.savefig(f"{prefix}_overlay.png", bbox_inches="tight", dpi=150)
    plt.close()

    # Union of instance masks (each already >= threshold) -> single binary mask.
    h, w = image.size[1], image.size[0]
    union = np.zeros((h, w), dtype=bool)
    for m in state["masks"]:
        union |= m.squeeze(0).cpu().numpy().astype(bool)
    Image.fromarray((union * 255).astype(np.uint8)).save(f"{prefix}_mask.png")
    print(f"wrote {prefix}_overlay.png and {prefix}_mask.png")

    # For WSIs, also dump polygon coordinates in full-res (level-0) WSI pixel space.
    if scale is not None:
        sx, sy = scale
        polys = []
        for m, sc in zip(state["masks"], state["scores"]):
            mask = m.squeeze(0).cpu().numpy().astype(np.uint8)
            # ponytail: RETR_EXTERNAL drops holes; use RETR_CCOMP if holes matter.
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                pts = c.reshape(-1, 2).astype(float) * (sx, sy)
                polys.append({"score": round(float(sc), 3), "polygon": pts.round().astype(int).tolist()})
        with open(f"{prefix}_coords.json", "w") as f:
            json.dump(polys, f)
        print(f"wrote {prefix}_coords.json ({len(polys)} polygon(s), level-0 coordinates)")


if __name__ == "__main__":
    main()
