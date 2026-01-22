# Base image: Python 3.12
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# --- Install uv ---
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# --- Install System Tools (Added unzip for Bun) ---
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    jq \
    iproute2 \
    unzip \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

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
