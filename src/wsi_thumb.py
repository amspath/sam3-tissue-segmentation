import argparse

from fastslide import FastSlide
from PIL import Image

TARGET_PX = 2048  # default thumbnail longest side when downsample is auto


def thumbnail(slide_path, downsample=None):
    """PIL RGB thumbnail + full-res (w, h). downsample=None auto-targets TARGET_PX longest side."""
    s = FastSlide.from_file_path(slide_path)
    w, h = s.dimensions
    if downsample is None:
        downsample = max(w, h) / TARGET_PX
    lvl = s.get_best_level_for_downsample(downsample)
    lw, lh = s.level_dimensions[lvl]
    img = Image.fromarray(s.read_region((0, 0), lvl, (lw, lh)).numpy()).convert("RGB")
    img = img.resize((round(w / downsample), round(h / downsample)), Image.LANCZOS)
    return img, (w, h)


def main():
    p = argparse.ArgumentParser(description="Make a downsampled thumbnail of a WSI.")
    p.add_argument("slide", help="Path to slide file")
    p.add_argument("-d", "--downsample", type=float, default=5.0, help="Downsample factor (default: 5)")
    p.add_argument("-o", "--output", help="Output PNG path (default: <slide>_thumb<D>x.png)")
    args = p.parse_args()

    img, (w, h) = thumbnail(args.slide, args.downsample)
    dest = args.output or f"{args.slide.rsplit('.', 1)[0]}_thumb{args.downsample:g}x.png"
    img.save(dest)
    print(f"{(w, h)} -> {img.size}  {dest}")


if __name__ == "__main__":
    main()
