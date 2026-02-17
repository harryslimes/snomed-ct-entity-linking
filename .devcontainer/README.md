# Devcontainer Profiles

This repo uses two local-build devcontainer profiles:

- `.devcontainer/amd64/devcontainer.json` for x86_64 NVIDIA hosts
- `.devcontainer/arm64/devcontainer.json` for ARM64 DGX Spark hosts

## How to choose in VS Code

1. Run `Dev Containers: Open Folder in Container...`
2. Select the profile that matches your machine architecture (`amd64` or `arm64`)

VS Code will then build the matching local Dockerfile:

- `amd64` profile -> `.devcontainer/Dockerfile.amd64`
- `arm64` profile -> `.devcontainer/Dockerfile.arm64`
