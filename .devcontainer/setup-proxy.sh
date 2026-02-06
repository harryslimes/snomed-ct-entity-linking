#!/usr/bin/env bash

if [[ "${1:-}" == "--install" ]]; then
  src="${BASH_SOURCE[0]}"
  if command -v sudo >/dev/null 2>&1; then
    sudo install -m 0755 "$src" /etc/profile.d/sidecar-proxy.sh
    exit 0
  fi

  install -m 0755 "$src" "$HOME/.config/sidecar-proxy.sh"

  for file in "$HOME/.bashrc" "$HOME/.profile"; do
    if [[ -f "$file" ]] && ! grep -q "sidecar-proxy.sh" "$file"; then
      printf '\n. "$HOME/.config/sidecar-proxy.sh"\n' >> "$file"
    fi
  done
  exit 0
fi

proxy_host="${SIDECAR_PROXY_HOST:-env-sidecar}"
proxy_port="${SIDECAR_PROXY_PORT:-8888}"
proxy_fallback="${SIDECAR_PROXY_FALLBACK:-1}"

case "${proxy_fallback,,}" in
  0|false|no|off)
    proxy_fallback=0
    ;;
  *)
    proxy_fallback=1
    ;;
esac

if [[ -n "${SIDECAR_PROXY_USERNAME:-}" && -n "${SIDECAR_PROXY_PASSWORD:-}" ]]; then
  proxy_url="http://${SIDECAR_PROXY_USERNAME}:${SIDECAR_PROXY_PASSWORD}@${proxy_host}:${proxy_port}"
else
  proxy_url="http://${proxy_host}:${proxy_port}"
fi

if [[ "$proxy_fallback" == "0" ]]; then
  export http_proxy="$proxy_url"
  export https_proxy="$proxy_url"
  export HTTP_PROXY="$proxy_url"
  export HTTPS_PROXY="$proxy_url"
  export ALL_PROXY="$proxy_url"

  base_no_proxy="localhost,127.0.0.1,${proxy_host}"
  if [[ -n "${NO_PROXY:-}" ]]; then
    export NO_PROXY="${base_no_proxy},${NO_PROXY}"
  else
    export NO_PROXY="$base_no_proxy"
  fi
  export no_proxy="$NO_PROXY"
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
  fi
  exit 0
fi

proxy_reachable=1
if ! getent hosts "$proxy_host" >/dev/null 2>&1; then
  proxy_reachable=0
elif command -v timeout >/dev/null 2>&1; then
  if ! timeout 1 bash -c ">/dev/tcp/${proxy_host}/${proxy_port}" >/dev/null 2>&1; then
    proxy_reachable=0
  fi
else
  if ! bash -c ">/dev/tcp/${proxy_host}/${proxy_port}" >/dev/null 2>&1; then
    proxy_reachable=0
  fi
fi

if [[ "$proxy_reachable" == "1" ]]; then
  export http_proxy="$proxy_url"
  export https_proxy="$proxy_url"
  export HTTP_PROXY="$proxy_url"
  export HTTPS_PROXY="$proxy_url"
  export ALL_PROXY="$proxy_url"

  base_no_proxy="localhost,127.0.0.1,${proxy_host}"
  if [[ -n "${NO_PROXY:-}" ]]; then
    export NO_PROXY="${base_no_proxy},${NO_PROXY}"
  else
    export NO_PROXY="$base_no_proxy"
  fi
  export no_proxy="$NO_PROXY"
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY no_proxy
fi
