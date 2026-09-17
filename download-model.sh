#!/bin/bash
# Legacy wrapper: downloads Qodo-Embed-1-1.5B. Prefer ./download-embed-model.sh [KEY].
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download-embed-model.sh" qodo-embed-1.5b
