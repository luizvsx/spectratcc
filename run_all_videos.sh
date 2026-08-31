#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$PROJECT_DIR/data/videos"
OUTPUT_DIR="$PROJECT_DIR/outputs"

mkdir -p $OUTPUT_DIR

for video in $INPUT_DIR/*.mp4; do
    nome=$(basename "$video")
    
    echo "Processando $nome..."
    
    python "$PROJECT_DIR/main.py" \
        --source_video_path "$video" \
        --target_video_path "$OUTPUT_DIR/out_$nome" \
        --mode FULL_ANALYSIS \
        --device cuda

    echo "Finalizado $nome"
done

echo "Todos os vídeos foram processados!"#!/bin/bash
