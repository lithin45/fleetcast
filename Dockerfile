# FleetCast image — single image used by both Compose services (pipeline + app).
# Built on the official uv image for reproducible, fast dependency installs.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# uv: copy (not hardlink) into the layer, compile bytecode for faster startup.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Runtime system deps: libgomp1 for LightGBM's OpenMP. GeoPandas needs no system
# GDAL because pyogrio ships GDAL-bundled wheels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install dependencies first (cached unless pyproject/uv.lock change).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Install the project itself.
COPY src ./src
COPY config ./config
COPY sql ./sql
COPY README.md ./README.md
RUN uv sync --frozen --no-dev

EXPOSE 8501

# Default to the CLI help; Compose overrides `command` per service.
CMD ["fleetcast", "--help"]
