# CPU image: runs tests, the smoke pipeline, evaluation, and quantization.
# For GPU training, swap the base for a CUDA image and install the matching torch wheel.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Torch CPU first (biggest layer, best cache hit rate)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY configs ./configs
COPY tests ./tests

# Hermetic verification at build time keeps the image honest
RUN python -c "import emocapnet; print(emocapnet.__version__)"

ENTRYPOINT ["emocapnet"]
CMD ["--help"]
