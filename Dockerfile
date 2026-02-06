# =============================================================================
# DGX Spark (GB10 / sm_121a) Devcontainer
# =============================================================================
# Base: lmsysorg/sglang:spark — the ONLY image with pre-compiled sm_121a
# kernels, NVFP4 support, and all the Triton/CUTLASS/PyTorch workarounds
# needed for GB10. Do NOT try to build SGLang from source for this hardware.
# =============================================================================
FROM lmsysorg/sglang:spark

# --- Install System Tools ---
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    jq iproute2 sudo unzip git kmod rsync curl ca-certificates \
    && mkdir -p /etc/sudoers.d \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# --- Create the vscode user ---
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN if getent passwd $USER_UID >/dev/null; then \
        EXISTING_USER=$(getent passwd $USER_UID | cut -d: -f1); \
        usermod -l $USERNAME $EXISTING_USER; \
        groupmod -n $USERNAME $(getent group $USER_GID | cut -d: -f1) || true; \
        usermod -d /home/$USERNAME -m $USERNAME; \
    else \
        if ! getent group $USER_GID >/dev/null; then \
            groupadd --gid $USER_GID $USERNAME; \
        fi; \
        useradd --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# --- Triton ptxas fix for sm_121a ---
# Ensures Triton uses the system ptxas (which knows about sm_121a)
# rather than its own bundled version.
RUN TRITON_PTXAS=$(find / -path "*/triton/backends/nvidia/bin/ptxas" 2>/dev/null | head -1) && \
    if [ -n "$TRITON_PTXAS" ]; then \
        ln -sf /usr/local/cuda/bin/ptxas "$TRITON_PTXAS"; \
        echo "Linked Triton ptxas -> /usr/local/cuda/bin/ptxas"; \
    fi
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# --- Install fastsafetensors for faster model loading ---
# Dramatically improves load times for large models on Spark's unified memory.
RUN pip install --no-deps "fastsafetensors>=0.1.10" 2>/dev/null || \
    echo "fastsafetensors install skipped (may need manual install)"

# --- Install Node.js & Bun ---
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

ENV BUN_INSTALL=/usr/local
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/usr/local/bin:${PATH}"

RUN bun install -g @anthropic-ai/claude-code

# --- Fix Permissions ---
USER vscode
RUN mkdir -p /home/vscode/.config/claude \
    && mkdir -p /home/vscode/.codex \
    && mkdir -p /home/vscode/.cache/huggingface \
    && mkdir -p /home/vscode/commandhistory \
    && touch /home/vscode/commandhistory/.bash_history