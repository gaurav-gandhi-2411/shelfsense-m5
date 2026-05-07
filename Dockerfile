# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Installs all Python deps and builds the shelfsense wheel into /app/.venv.
# Nothing from this stage reaches the runtime image except the venv.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl tzdata \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Pull uv binary directly from its official image (no install scripts)
COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /usr/local/bin/uv

WORKDIR /app

# Layer-cache dep install: only re-runs when pyproject.toml or uv.lock changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --python python3.12

# Install shelfsense as a non-editable wheel (source baked into site-packages)
COPY shelfsense/ ./shelfsense/
COPY README.md ./
RUN uv sync --frozen --no-editable --python python3.12


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# Slim image: Python 3.12 + pre-built venv copied from builder. No build tools.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates tzdata \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid 1000 shelfsense \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash shelfsense

WORKDIR /app
RUN chown shelfsense:shelfsense /app

# Pre-built venv from builder — contains all deps + shelfsense wheel
COPY --from=builder --chown=shelfsense:shelfsense /app/.venv /app/.venv

# Volume mount points: keep large/mutable data outside the image layer
VOLUME ["/app/data", "/app/mlruns", "/app/dagster_home"]

USER shelfsense

ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["shelfsense"]
CMD ["--help"]
