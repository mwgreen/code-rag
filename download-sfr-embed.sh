#!/bin/bash
# Legacy wrapper: downloads SFR-Embedding-Code-2B_R. Prefer ./download-embed-model.sh [KEY].
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download-embed-model.sh" sfr-embed-code-2b
