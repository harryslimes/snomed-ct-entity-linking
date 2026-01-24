# Base image: Python 3.12
FROM nvcr.io/nvidia/pytorch:25.12-py3
# --- Install uv ---
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# --- Install System Tools (Added unzip for Bun) ---
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    jq \
    iproute2 \
    sudo \
    unzip \
    && mkdir -p /etc/sudoers.d \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# --- Create the vscode user safely ---
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

# --- Install Node.js --- (Required by Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# --- Install Bun & Claude Code ---
# 1. Install Bun explicitly to /usr/local so it is available to ALL users
ENV BUN_INSTALL=/usr/local
RUN curl -fsSL https://bun.sh/install | bash

# 2. Add Bun to PATH globally
ENV PATH="/usr/local/bin:${PATH}"

# 3. Install Claude Code using Bun (Much faster, no npm hang)
RUN bun install -g @anthropic-ai/claude-code

# --- Fix Permissions for Volumes ---
# Switch to vscode to create folders with correct ownership
USER vscode

RUN mkdir -p /home/vscode/.config/claude \
    && mkdir -p /home/vscode/.codex \
    && mkdir -p /home/vscode/.cache/huggingface \
    && mkdir -p /home/vscode/commandhistory \
    # Extra safety: Ensure Bun cache doesn't cause permission issues if used later
    && mkdir -p /home/vscode/.bun

# Configure Bash to sync history immediately and across multiple terminals
RUN echo 'export PROMPT_COMMAND="history -a; history -c; history -r; $PROMPT_COMMAND"' >> /home/vscode/.bashrc \
    && echo 'shopt -s histappend' >> /home/vscode/.bashrc
