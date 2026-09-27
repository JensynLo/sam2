# SAM 2 web UI (app.py) on GPU.
#
#   docker compose up -d --build        # see README.md, "Docker"
#
# Checkpoints are not baked into the image: mount them at /app/checkpoints.

# PyTorch 2.11 + CUDA 12.8 (required by RTX 50-series / Blackwell GPUs)
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # the image's PyTorch lives in the system Python 3.12 (Ubuntu 24.04), which is
    # marked externally managed (PEP 668); installing next to it is intended here
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    # no nvcc in the runtime image: skip the optional CUDA post-processing extension
    SAM2_BUILD_CUDA=0 \
    # the container may run as an arbitrary non-root user (see docker-compose.yaml)
    HOME=/tmp

WORKDIR /app

# Install the sam2 package and its dependencies first, so that changes to the
# app code below don't invalidate this layer. `--no-build-isolation` reuses the
# image's PyTorch instead of downloading another copy just to build.
COPY setup.py pyproject.toml MANIFEST.in README.md ./
COPY sam2 ./sam2
RUN python -m pip install --no-build-isolation -e ".[web]"

COPY infer.py app.py ./
COPY web ./web
RUN mkdir -p checkpoints outputs && chmod 777 outputs

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/info', timeout=4)"

# Inside the container, listen on all interfaces; which host addresses can
# reach it is decided by the port mapping (127.0.0.1 only by default).
ENTRYPOINT ["python", "app.py", "--host", "0.0.0.0", "--output-dir", "/app/outputs"]
