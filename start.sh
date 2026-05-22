#!/usr/bin/env bash
# start.sh — boot backend (uvicorn :8000) and frontend (vite :5173) for OrderingServiceText.

set -u
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p /tmp/ost_logs

# Backend
if command -v lsof >/dev/null 2>&1; then
  lsof -ti tcp:8000 | xargs -r kill -9 2>/dev/null || true
  lsof -ti tcp:5173 | xargs -r kill -9 2>/dev/null || true
fi

# Prefer system python3 because the project's myvenv often has a stale dyld link.
PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ] && [ -x "${PROJECT_DIR}/myvenv/bin/python" ]; then
  PY="${PROJECT_DIR}/myvenv/bin/python"
fi

echo "[start.sh] backend: $PY -m uvicorn app:app --host 0.0.0.0 --port 8000"
nohup "$PY" -m uvicorn app:app --host 0.0.0.0 --port 8000 \
  > /tmp/ost_logs/backend.log 2>&1 &
echo $! > /tmp/ost_logs/backend.pid

# Frontend
cd "$PROJECT_DIR/frontend"
echo "[start.sh] frontend: npm run dev (port 5173)"
nohup npm run dev -- --host 0.0.0.0 --port 5173 \
  > /tmp/ost_logs/frontend.log 2>&1 &
echo $! > /tmp/ost_logs/frontend.pid

echo "[start.sh] backend pid=$(cat /tmp/ost_logs/backend.pid 2>/dev/null) frontend pid=$(cat /tmp/ost_logs/frontend.pid 2>/dev/null)"
echo "[start.sh] logs: /tmp/ost_logs/backend.log /tmp/ost_logs/frontend.log"
