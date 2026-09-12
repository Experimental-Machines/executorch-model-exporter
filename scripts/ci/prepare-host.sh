#!/usr/bin/env bash
# Makes a GitHub-hosted Ubuntu runner fit for exporting a multi-GB model:
#   - deletes preinstalled toolchains the export never uses,
#   - picks the filesystem with the most free space as the work directory,
#   - replaces the image's small swap with a larger swap file.
#
#   prepare-host.sh <swap-gib>
#
# Exports WORK_DIR and HF_HOME through $GITHUB_ENV (or prints them outside Actions).
set -euo pipefail

SWAP_GIB="${1:-16}"

show() {
  echo "::group::$1"
  echo "nproc: $(nproc)"
  free -b
  df -B1 / /mnt 2>/dev/null || df -B1 /
  swapon --show --bytes || true
  echo "::endgroup::"
}

free_bytes() { df -B1 --output=avail "$1" | tail -1; }

show "host before cleanup"

sudo rm -rf \
  /usr/share/dotnet /usr/local/lib/android /opt/ghc /usr/local/.ghcup \
  /opt/hostedtoolcache/CodeQL /usr/share/swift /usr/local/share/boost \
  /usr/local/share/chromium /usr/local/share/powershell /opt/microsoft /opt/google \
  /usr/local/julia* 2>/dev/null || true
sudo docker image prune --all --force >/dev/null 2>&1 || true

root_free=$(free_bytes /)
mnt_free=0
if mountpoint -q /mnt; then mnt_free=$(free_bytes /mnt); fi
if [ "$mnt_free" -gt "$root_free" ]; then
  work_base=/mnt
  other_base=/
else
  work_base=/
  other_base=/mnt
fi
work_dir="${work_base%/}/exe-expo-work"
sudo mkdir -p "$work_dir"
sudo chown "$(id -u):$(id -g)" "$work_dir"

# Swap goes on the filesystem the export is not writing to, when it has room for it.
swap_bytes=$((SWAP_GIB * 1024 * 1024 * 1024))
margin=$((2 * 1024 * 1024 * 1024))
swap_base="$work_base"
if [ "$other_base" = "/mnt" ] && mountpoint -q /mnt && [ "$(free_bytes /mnt)" -gt $((swap_bytes + margin)) ]; then
  swap_base=/mnt
elif [ "$other_base" = "/" ] && [ "$(free_bytes /)" -gt $((swap_bytes + margin)) ]; then
  swap_base=/
fi
sudo swapoff -a || true
sudo rm -f /mnt/swapfile /swapfile
swap_file="${swap_base%/}/exe-expo.swap"
sudo fallocate -l "${SWAP_GIB}G" "$swap_file"
sudo chmod 600 "$swap_file"
sudo mkswap "$swap_file" >/dev/null
sudo swapon "$swap_file"

show "host after cleanup (work in $work_dir, swap in $swap_file)"

if [ -n "${GITHUB_ENV:-}" ]; then
  echo "WORK_DIR=$work_dir" >>"$GITHUB_ENV"
  echo "HF_HOME=$work_dir/hf-home" >>"$GITHUB_ENV"
else
  echo "WORK_DIR=$work_dir"
  echo "HF_HOME=$work_dir/hf-home"
fi
