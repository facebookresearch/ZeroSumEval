#!/bin/bash
set -e

python paper/visualize_stats.py
python paper/visualize_llama.py
python paper/visualize_predict_vs_cot.py
python paper/visualize_reasoning_vs_cot.py
python paper/visualize_cumulative_ratings.py

echo "Figures generated successfully."