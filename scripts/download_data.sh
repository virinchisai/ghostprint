#!/usr/bin/env bash
# Fetch LibriSpeech dev-clean (train side) and test-clean (eval side), ~680 MB.
set -euo pipefail
cd "$(dirname "$0")/../data"
for part in dev-clean test-clean; do
  [ -d "LibriSpeech/$part" ] && { echo "$part present"; continue; }
  curl -L -O "https://www.openslr.org/resources/12/$part.tar.gz"
  tar xzf "$part.tar.gz" && rm "$part.tar.gz"
done
