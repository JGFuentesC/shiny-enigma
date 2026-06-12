#!/usr/bin/env bash
# run_pipeline.sh - Execute the full pipeline on platypy server
# Usage: ./run_pipeline.sh [step]
#   step: 1|2|3|4|train|all (default: all)

STEP=${1:-all}

echo "=== Anime Faces Pipeline ==="
echo "Step: $STEP"
echo ""

if [ "$STEP" = "all" ] || [ "$STEP" = "1" ]; then
    echo ">>> Step 1: Extract sample + classify with gemma4"
    python3 /workspace/ssd/solucion_practica/cluster/01_extract_sample.py
    echo ""
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "2" ]; then
    echo ">>> Step 2: Consolidate categories"
    python3 /workspace/ssd/solucion_practica/cluster/02_consolidate_categories.py
    echo ""
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "3" ]; then
    echo ">>> Step 3: Re-classify sample in 4 categories"
    python3 /workspace/ssd/solucion_practica/cluster/03_reclassify_sample.py
    echo ""
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "4" ]; then
    echo ">>> Step 4: Extrapolate labels to full dataset"
    python3 /workspace/ssd/solucion_practica/cluster/04_extrapolate.py
    echo ""
fi

if [ "$STEP" = "all" ] || [ "$STEP" = "train" ]; then
    echo ">>> Train CNN from scratch"
    python3 /workspace/ssd/solucion_practica/cnn/train.py
    echo ""
fi

echo "=== Pipeline complete ==="
