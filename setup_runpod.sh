#!/bin/bash
# Setup script for RunPod instances
# Recommended template: Official Parameter Golf template
#   https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th
# If using a custom template, use pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
#
# For a feasibility test: 1x H100 ($3-4/hr)
# For full search: 1x H100 or 1x A100 80GB ($2-3/hr)
# For final submission: 8x H100 SXM (required by challenge rules)
#
# Usage:
#   ssh into your runpod, then:
#   cd /workspace
#   git clone <your-repo-url> parameter-golf
#   cd parameter-golf
#   bash setup_runpod.sh

set -e

echo "=== Setting up NeuroMod Evolutionary Architecture Search ==="

# Install dependencies (if not using the official template)
pip install --quiet numpy tqdm torch huggingface-hub sentencepiece datasets tiktoken matplotlib 2>/dev/null || true

# Download FineWeb data (1 shard for quick testing, 80 for full runs)
SHARDS=${1:-1}
echo ""
echo "=== Downloading FineWeb dataset (${SHARDS} training shard(s)) ==="
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$SHARDS"

echo ""
echo "=== Dataset ready ==="
ls -lh data/datasets/fineweb10B_sp1024/
ls -lh data/tokenizers/

echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU"

echo ""
echo "=== Quick Commands ==="
echo ""
echo "# 1. Quick feasibility test (single config, ~5 min on 1x H100):"
echo "python -m neuromod_recursive.run_search --single --use-fineweb --steps 5000 --seq-len 1024"
echo ""
echo "# 2. Mini evolutionary search (~2 hours on 1x H100):"
echo "python -m neuromod_recursive.run_search --use-fineweb --population 10 --generations 5 --steps 1000 --seq-len 1024"
echo ""
echo "# 3. Full evolutionary search (~10-20 hours on 1x H100):"
echo "python -m neuromod_recursive.run_search --use-fineweb --population 30 --generations 20 --steps 2000 --seq-len 1024"
echo ""
echo "# 4. Download more training shards (for longer runs):"
echo "python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10"
echo ""
echo "=== Setup complete! ==="
