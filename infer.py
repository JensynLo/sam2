# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Segment an object in an image with SAM 2 from point / box prompts.

Examples:
  # one foreground click
  python infer.py notebooks/images/truck.jpg --point 500,375
  # foreground + background clicks (label 1 = foreground, 0 = background)
  python infer.py notebooks/images/truck.jpg --point 500,375 --point 1125,625,0
  # a box, optionally refined with clicks
  python infer.py notebooks/images/truck.jpg --box 425,600,700,875 --point 575,750,0
"""

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

REPO_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = REPO_DIR / "checkpoints"

# model name -> (hydra config inside the sam2 package, checkpoint file name),
# ordered from the most to the least accurate one
MODELS = {
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
}
OUTPUT_KINDS = ("mask", "overlay", "cutout")
MASK_COLOR = np.array([30, 144, 255], dtype=np.float32)


def get_device(device=None):
    """Return the device to run on: CUDA unless another one is asked for explicitly.

    There is no silent fallback to CPU: if CUDA is unavailable, fail loudly.
    """
    device = torch.device(device or "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available (check the GPU driver and that PyTorch is a CUDA "
            "build); pass --device cpu to run on CPU explicitly"
        )
    return device


def resolve_model(model=None, checkpoint=None):
    """Return (model name, config, checkpoint path) for the requested model.

    Without an explicit model, use the most accurate one found in `checkpoints/`.
    """
    if model is None:
        if checkpoint is not None:
            raise ValueError(
                "--checkpoint requires --model to pick the matching config"
            )
        model = next(
            (
                name
                for name, (_, ckpt) in MODELS.items()
                if (CHECKPOINT_DIR / ckpt).is_file()
            ),
            None,
        )
        if model is None:
            raise FileNotFoundError(
                f"No SAM 2.1 checkpoint found in {CHECKPOINT_DIR}. Download one first, e.g.\n"
                "  cd checkpoints && wget https://dl.fbaipublicfiles.com/"
                "segment_anything_2/092824/sam2.1_hiera_large.pt"
            )
    config, ckpt_name = MODELS[model]
    ckpt_path = Path(checkpoint) if checkpoint else CHECKPOINT_DIR / ckpt_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return model, config, ckpt_path


class Segmenter:
    """A SAM 2 image predictor that returns one mask per set of prompts."""

    def __init__(self, model=None, checkpoint=None, device=None):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = get_device(device)
        self.model_name, config, ckpt_path = resolve_model(model, checkpoint)
        if (
            self.device.type == "cuda"
            and torch.cuda.get_device_properties(0).major >= 8
        ):
            # turn on tfloat32 for Ampere GPUs and newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        sam2_model = build_sam2(config, str(ckpt_path), device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def set_image(self, image):
        """Compute the image embedding; `image` is an HxWx3 uint8 RGB array."""
        with torch.inference_mode(), self._autocast():
            self.predictor.set_image(image)

    def predict(self, points=(), box=None):
        """
        Segment the object described by the prompts on the current image.

        - points: sequence of (x, y, label) in pixels, label 1 = foreground, 0 = background
        - box: (x0, y0, x1, y1) in pixels, or None

        Returns an HxW bool mask and the model's quality score for it.
        """
        if len(points) == 0 and box is None:
            raise ValueError("At least one point or a box is required")
        point_coords = point_labels = None
        if len(points) > 0:
            points = np.asarray(points, dtype=np.float32)
            point_coords, point_labels = points[:, :2], points[:, 2].astype(np.int32)
        # A single click is ambiguous (e.g. part vs. whole object), so ask for three
        # candidate masks and keep the best-scored one, as recommended by SAM.
        multimask_output = len(points) == 1 and box is None
        with torch.inference_mode(), self._autocast():
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None if box is None else np.asarray(box, dtype=np.float32),
                multimask_output=multimask_output,
            )
        best = int(np.argmax(scores))
        return masks[best].astype(bool), float(scores[best])


def load_image(fp):
    """Load an image (path or file object) as an HxWx3 uint8 RGB array."""
    with Image.open(fp) as img:
        # apply the EXIF orientation so that pixel coordinates match what viewers show
        return np.array(ImageOps.exif_transpose(img).convert("RGB"))


def render(image, mask, kind):
    """
    Render a segmentation result as a PIL image:
    - "mask": black/white mask
    - "overlay": the image with the mask highlighted
    - "cutout": the image with a transparent background outside the mask
    """
    if kind == "mask":
        return Image.fromarray(mask.astype(np.uint8) * 255)
    if kind == "overlay":
        out = image.astype(np.float32)
        out[mask] = out[mask] * 0.45 + MASK_COLOR * 0.55
        return Image.fromarray(out.round().astype(np.uint8))
    if kind == "cutout":
        return Image.fromarray(np.dstack([image, mask.astype(np.uint8) * 255]))
    raise ValueError(f"Unknown output kind {kind!r}, expected one of {OUTPUT_KINDS}")


def _numbers(text, count, name):
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        values = []
    if len(values) not in count:
        raise argparse.ArgumentTypeError(f"invalid {name}: {text!r}")
    return values


def parse_point(text):
    values = _numbers(text, (2, 3), "point, expected X,Y or X,Y,LABEL")
    x, y, label = values if len(values) == 3 else values + [1]
    if label not in (0, 1):
        raise argparse.ArgumentTypeError(f"point label must be 0 or 1: {text!r}")
    return x, y, int(label)


def parse_box(text):
    x0, y0, x1, y1 = _numbers(text, (4,), "box, expected X0,Y0,X1,Y1")
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def parse_kinds(text):
    kinds = [k.strip() for k in text.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in OUTPUT_KINDS]
    if not kinds or unknown:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated subset of {','.join(OUTPUT_KINDS)}: {text!r}"
        )
    return kinds


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Segment an object in an image with SAM 2 from point / box prompts.",
        epilog="Examples:" + __doc__.split("Examples:", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", type=Path, help="input image")
    parser.add_argument(
        "--point",
        "-p",
        type=parse_point,
        action="append",
        default=[],
        metavar="X,Y[,LABEL]",
        help="point prompt in pixels; LABEL is 1 (foreground, default) or 0 (background); repeatable",
    )
    parser.add_argument(
        "--box",
        "-b",
        type=parse_box,
        metavar="X0,Y0,X1,Y1",
        help="box prompt in pixels",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("outputs"),
        help="output directory (default: outputs)",
    )
    parser.add_argument(
        "--save",
        type=parse_kinds,
        default=["mask", "overlay"],
        metavar="KINDS",
        help=f"comma-separated outputs among {','.join(OUTPUT_KINDS)} (default: mask,overlay)",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        help="model size (default: the largest one found in checkpoints/)",
    )
    parser.add_argument(
        "--checkpoint", help="checkpoint path (overrides the default for --model)"
    )
    parser.add_argument(
        "--device", help="cuda (default; fails if unavailable), cuda:N or cpu"
    )
    args = parser.parse_args(argv)

    if not args.point and args.box is None:
        parser.error("give at least one --point or a --box")
    try:
        image = load_image(args.image)
    except OSError as e:
        parser.error(f"cannot read image {args.image}: {e}")
    try:
        segmenter = Segmenter(args.model, args.checkpoint, args.device)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        sys.exit(f"error: {e}")
    print(f"model: sam2.1_{segmenter.model_name}, device: {segmenter.device}")

    segmenter.set_image(image)
    mask, score = segmenter.predict(args.point, args.box)
    print(f"score: {score:.3f}, mask area: {mask.mean():.1%} of the image")

    args.output.mkdir(parents=True, exist_ok=True)
    for kind in args.save:
        path = args.output / f"{args.image.stem}_{kind}.png"
        render(image, mask, kind).save(path)
        print(f"saved {kind}: {path}")


if __name__ == "__main__":
    main()
