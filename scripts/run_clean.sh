#!/usr/bin/env bash
set -euo pipefail
python -m src.train --config experiments/configs/imdb_cnn.yaml
python -m src.eval  --config experiments/configs/imdb_cnn.yaml --ckpt experiments/logs/imdb_cnn.pt