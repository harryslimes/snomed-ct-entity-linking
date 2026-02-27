#!/usr/bin/env bash
set -euo pipefail

DEVCONT=env-sidecar-devcontainer
WORKDIR=/workspaces/env-sidecar-dev

ensure_devcontainer_running() {
  if docker ps -q -f "name=^${DEVCONT}$" | grep -q .; then
    return 0
  fi

  if docker ps -aq -f "name=^${DEVCONT}$" | grep -q .; then
    echo "Starting dev container: ${DEVCONT}"
    docker start "${DEVCONT}" >/dev/null
  else
    if command -v devcontainer >/dev/null 2>&1; then
      if [ ! -d "${WORKDIR}" ]; then
        echo "Missing workspace at ${WORKDIR}; set WORKDIR to your host path." >&2
        exit 1
      fi
      echo "Creating dev container: ${DEVCONT}"
      devcontainer up --workspace-folder "${WORKDIR}" >/dev/null
    else
      echo "Dev container ${DEVCONT} not found. Start it with VS Code Dev Containers or install the devcontainer CLI." >&2
      exit 1
    fi
  fi

  for _ in $(seq 1 30); do
    if docker ps -q -f "name=^${DEVCONT}$" | grep -q .; then
      return 0
    fi
    sleep 1
  done

  echo "Dev container ${DEVCONT} did not start in time." >&2
  exit 1
}

ensure_devcontainer_running

docker stop env-sidecar || true
docker rm env-sidecar || true

docker exec "${DEVCONT}" bash -lc "cd ${WORKDIR} && mkdir -p build && go build -o build/env-sidecar ."
docker cp "${DEVCONT}:${WORKDIR}/build/env-sidecar" ./env-sidecar

docker compose up -d --build
docker compose logs -f env-sidecar
