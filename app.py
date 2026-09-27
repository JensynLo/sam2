# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A local web UI for SAM 2: drop an image, click / draw a box to segment an object,
preview the mask and save it.

  python app.py                   # then open http://127.0.0.1:7860
  python app.py --host tailscale  # reachable from the tailnet, e.g. http://<machine>:7860
"""

import argparse
import io
import json
import math
import subprocess
import sys
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from flask import abort, Flask, jsonify, request, send_file, send_from_directory

from infer import load_image, MASK_COLOR, MODELS, OUTPUT_KINDS, render, Segmenter
from PIL import Image
from werkzeug.exceptions import HTTPException

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_CACHED_IMAGES = 8


def _png_response(img, download_name=None):
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=download_name is not None,
        download_name=download_name,
    )


def _json_object():
    data = request.get_json(silent=True)
    if data is None:
        return {}
    if not isinstance(data, dict):
        abort(400, "请求体必须是 JSON 对象")
    return data


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _coord(v):
    if not (_is_int(v) or isinstance(v, float)) or not math.isfinite(v):
        raise ValueError(v)
    return float(v)


def _parse_prompts(data):
    """Validate {"points": [[x, y, label], ...], "box": [x0, y0, x1, y1] | null}."""
    points, box = data.get("points", []), data.get("box")
    try:
        if not isinstance(points, list):
            raise ValueError(points)
        parsed_points = []
        for point in points:
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError(point)
            x, y, label = point
            if not _is_int(label) or label not in (0, 1):
                raise ValueError(label)
            parsed_points.append((_coord(x), _coord(y), label))
        if box is not None:
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(box)
            x0, y0, x1, y1 = (_coord(v) for v in box)
            box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    except ValueError:
        abort(
            400, "提示格式错误:点为 [x, y, 0|1],框为 [x0, y0, x1, y1],坐标须为有限数值"
        )
    if not parsed_points and box is None:
        abort(400, "至少需要一个点或一个框")
    return parsed_points, box


def _parse_version(value):
    """Parse an optional client-side result version (a non-negative integer)."""
    if value is None:
        return None
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if not _is_int(value) or value < 0:
        abort(400, "version 必须是非负整数")
    return value


def create_app(segmenter, output_dir):
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

    # The predictor holds the embedding of one image at a time and isn't thread-safe.
    lock = threading.Lock()
    # image id -> {"image": HxWx3 array, "name": file stem, "result": (version, mask)}
    # where "result" is the prediction with the highest version seen so far (or None)
    images = OrderedDict()
    current = {"id": None}

    def get_entry(image_id):
        entry = images.get(image_id)
        if entry is None:
            abort(404, "图片已过期,请重新上传")
        images.move_to_end(image_id)
        return entry

    @app.errorhandler(HTTPException)
    def handle_http_error(e):
        return jsonify(error=e.description), e.code

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/api/info")
    def info():
        return jsonify(
            model=f"sam2.1_{segmenter.model_name}",
            device=str(segmenter.device),
            output_dir=str(output_dir),
            kinds=OUTPUT_KINDS,
        )

    @app.post("/api/images")
    def upload_image():
        file = request.files.get("image")
        if file is None:
            abort(400, "没有收到图片")
        try:
            image = load_image(file.stream)
        except (OSError, Image.DecompressionBombError):
            abort(400, "无法识别的图片格式")
        image_id = uuid.uuid4().hex
        with lock:
            segmenter.set_image(image)
            current["id"] = image_id
            images[image_id] = {
                "image": image,
                "name": Path(file.filename or "image").stem or "image",
                "result": None,
            }
            while len(images) > MAX_CACHED_IMAGES:
                images.popitem(last=False)
        return jsonify(
            id=image_id,
            name=images[image_id]["name"],
            width=image.shape[1],
            height=image.shape[0],
        )

    @app.get("/api/images/<image_id>")
    def get_image(image_id):
        # Serve the decoded image so the browser shows exactly the pixels the model sees
        # (e.g. with the EXIF orientation applied).
        buf = io.BytesIO()
        Image.fromarray(get_entry(image_id)["image"]).save(
            buf, format="JPEG", quality=92
        )
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")

    @app.post("/api/images/<image_id>/predict")
    def predict(image_id):
        data = _json_object()
        points, box = _parse_prompts(data)
        # Requests may run out of order (the server is threaded), so the client numbers
        # them and only the newest result is kept for exporting/saving. Without a
        # version, a request counts as newer than any before it.
        version = _parse_version(data.get("version"))
        with lock:
            entry = get_entry(image_id)
            if current["id"] != image_id:
                segmenter.set_image(entry["image"])
                current["id"] = image_id
            mask, score = segmenter.predict(points, box)
            latest = entry["result"][0] if entry["result"] else -1
            if version is None:
                version = latest + 1
            if version > latest:
                entry["result"] = (version, mask)
        # a colored, transparent-outside mask layer to draw over the image
        layer = np.zeros((*mask.shape, 4), dtype=np.uint8)
        layer[mask] = (*MASK_COLOR.astype(np.uint8), 255)
        response = _png_response(Image.fromarray(layer))
        response.headers["X-Mask-Score"] = f"{score:.4f}"
        response.headers["X-Mask-Area"] = f"{mask.mean():.6f}"
        response.headers["X-Mask-Version"] = str(version)
        return response

    def rendered_result(image_id, kind, version):
        """Render the kept result; if `version` is given, it must be the kept one."""
        if kind not in OUTPUT_KINDS:
            abort(400, f"未知的保存类型: {kind}")
        entry = get_entry(image_id)
        result = entry["result"]
        if result is None:
            abort(400, "还没有分割结果")
        if version is not None and version != result[0]:
            abort(409, "要保存的结果已不是最新结果,请等待分割完成后再保存")
        return entry, render(entry["image"], result[1], kind)

    @app.get("/api/images/<image_id>/export/<kind>")
    def export(image_id, kind):
        version = _parse_version(request.args.get("version"))
        entry, img = rendered_result(image_id, kind, version)
        return _png_response(img, download_name=f"{entry['name']}_{kind}.png")

    @app.post("/api/images/<image_id>/save")
    def save(image_id):
        data = _json_object()
        kind = data.get("kind", "mask")
        if not isinstance(kind, str):
            abort(400, "kind 必须是字符串")
        entry, img = rendered_result(
            image_id, kind, _parse_version(data.get("version"))
        )

        raw_path = data.get("path") or ""
        if not isinstance(raw_path, str):
            abort(400, "path 必须是字符串")
        raw_path = raw_path.strip()
        path = Path(raw_path).expanduser() if raw_path else output_dir
        if not path.is_absolute():
            path = output_dir / path
        if not raw_path or raw_path.endswith(("/", "\\")) or path.is_dir():
            path = path / f"{entry['name']}_{kind}.png"
        elif not path.suffix:
            path = path.with_suffix(".png")
        if kind == "cutout" and path.suffix.lower() not in (".png", ".webp"):
            abort(400, "抠图带透明通道,请保存为 .png")
        if path.exists() and not data.get("overwrite"):
            return jsonify(error="文件已存在", exists=True, path=str(path)), 409

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if img.mode != "RGB" and path.suffix.lower() in (".jpg", ".jpeg"):
                img = img.convert("RGB")
            img.save(path)
        except (OSError, ValueError, KeyError) as e:
            abort(400, f"保存失败: {e}")
        return jsonify(path=str(path.resolve()))

    return app


def _tailscale_self():
    """Return (IPv4, MagicDNS name) of this machine on the tailnet, or None."""
    try:
        out = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        me = json.loads(out)["Self"]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return None
    ipv4 = next((ip for ip in me.get("TailscaleIPs") or [] if "." in ip), None)
    if ipv4 is None or not me.get("Online", True):
        return None
    return ipv4, (me.get("DNSName") or "").rstrip(".")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (default: 127.0.0.1); 'tailscale' binds to this machine's "
        "Tailscale IP so that only the tailnet can reach it. The UI can write files on "
        "this machine, so only expose it on networks you trust.",
    )
    parser.add_argument("--port", type=int, default=7860, help="port (default: 7860)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="default save directory; relative save paths are resolved against it (default: outputs)",
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

    tailnet = None
    if args.host in ("tailscale", "0.0.0.0", "::"):
        tailnet = _tailscale_self()
        if args.host == "tailscale":
            if tailnet is None:
                sys.exit(
                    "error: cannot get this machine's Tailscale IP; is Tailscale "
                    "installed and up (`tailscale status`)?"
                )
            args.host = tailnet[0]

    try:
        segmenter = Segmenter(args.model, args.checkpoint, args.device)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        sys.exit(f"error: {e}")
    print(f"model: sam2.1_{segmenter.model_name}, device: {segmenter.device}")
    app = create_app(segmenter, args.output_dir.expanduser().resolve())
    if tailnet and tailnet[1]:
        print(f"tailnet URL: http://{tailnet[1]}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
