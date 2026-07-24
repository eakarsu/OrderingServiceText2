#!/usr/bin/env bash
# Safe foreground launcher. It never kills processes, migrates, or seeds data.

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$PROJECT_DIR/.env" ]; then
  echo "Missing .env" >&2
  exit 1
fi
set -a
. "$PROJECT_DIR/.env"
set +a
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
  default_python="$PROJECT_DIR/.venv/bin/python"
else
  default_python="$(command -v python3 || command -v python || true)"
fi
PYTHON_BIN="${PYTHON_BIN:-$default_python}"
export DB_DATABASE="${DB_DATABASE:-${DB_NAME:-${PGDATABASE:-}}}"

if [ -z "$PYTHON_BIN" ]; then
  echo "Python was not found. Create a virtual environment and install requirements.txt." >&2
  exit 1
fi
if command -v lsof >/dev/null 2>&1 && lsof -ti "tcp:${BACKEND_PORT}" >/dev/null 2>&1; then
  echo "Backend port ${BACKEND_PORT} is already in use; refusing to terminate that process." >&2
  exit 1
fi
if [ "${NODE_ENV:-}" != "test" ] && command -v lsof >/dev/null 2>&1 && lsof -ti "tcp:${FRONTEND_PORT}" >/dev/null 2>&1; then
  echo "Frontend port ${FRONTEND_PORT} is already in use; refusing to terminate that process." >&2
  exit 1
fi

cleanup() {
  if [ -n "${BACKEND_PID:-}" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$PROJECT_DIR"
if [ "${NODE_ENV:-}" = "test" ]; then
  exec "$PYTHON_BIN" -m uvicorn app:app --host 127.0.0.1 --port "$BACKEND_PORT"
fi
"$PYTHON_BIN" -m uvicorn app:app --host 127.0.0.1 --port "$BACKEND_PORT" &
BACKEND_PID=$!

cd "$PROJECT_DIR/frontend"
exec npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT" --strictPort
