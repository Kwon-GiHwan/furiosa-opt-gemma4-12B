#!/bin/bash
# README "Toolchain" 절차. CI 와 컨테이너가 공용으로 실행한다.
# x86_64 Linux 전용. 버전은 여기 한 곳에서만 관리한다.
set -euo pipefail

RUST_NIGHTLY="nightly-2026-05-01"
FURIOSA_OPT_VERSION="0.6.0"

if [ "$(uname -m)" != "x86_64" ] || [ "$(uname -s)" != "Linux" ]; then
    echo "install_toolchain.sh: x86_64 Linux only (got $(uname -s)/$(uname -m))" >&2
    exit 1
fi

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update
# libc6-dev-arm64-cross 는 gcc-aarch64-linux-gnu 의 Recommends 라 명시해야 한다.
# 없으면 #[device] 매크로의 C 컴파일이 bits/wordsize.h 를 찾지 못한다.
$SUDO apt-get install -y --no-install-recommends \
    build-essential libclang-dev gcc-aarch64-linux-gnu libc6-dev-arm64-cross \
    curl ca-certificates git python3 pkg-config

if ! command -v rustup >/dev/null; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --default-toolchain none
fi
export PATH="$HOME/.cargo/bin:$PATH"

rustup toolchain install "$RUST_NIGHTLY" --profile minimal --component rustfmt --component clippy
rustup default "$RUST_NIGHTLY"

if ! command -v cargo-binstall >/dev/null; then
    curl -L --proto "=https" --tlsv1.2 -sSf \
        https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash
fi
cargo binstall --no-confirm "cargo-furiosa-opt@$FURIOSA_OPT_VERSION"
cargo binstall --no-confirm furiosa-arena-cli
cargo binstall --no-confirm moa-submitter-cli

cargo furiosa-opt --version
