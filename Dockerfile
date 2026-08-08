# Multi-stage build. The runtime image carries no build toolchain and runs as a
# non-root user — a container that can write its own code is one exploit away
# from persistence.

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt


FROM python:3.12-slim AS runtime

# git and ripgrep are hard runtime dependencies: the repository tools shell out
# to both. Missing either turns every investigation into a tool error.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ripgrep ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WORKSPACE_ROOT=/data/workspaces \
    DATABASE_URL=sqlite:////data/app.db

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY . .

# Non-root, and /data is the only writable path.
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /data/workspaces \
 && chown -R appuser:appuser /data /app
USER appuser

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status==200 else 1)"

# Threads, not processes: the agent loop blocks on I/O (model calls, git,
# ripgrep) rather than CPU, and SSE holds a connection open for the whole run.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--threads", "8", \
     "--worker-class", "gthread", \
     "--timeout", "900", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
