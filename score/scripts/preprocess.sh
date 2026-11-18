#!/bin/bash
# Convert raw documents of one domain into 8K-token source documents with unique data_id.
# Windows are taken from the beginning, end, and intermediate positions of long documents.
# Usage: bash score/scripts/preprocess.sh arxiv|code

DOMAIN=${1:?"Usage: $0 arxiv|code"}

# Set input parameters
INPUT_FOLDER="data/${DOMAIN}/"                              # Input folder path
OUTPUT_FOLDER="data/law_${DOMAIN}/segments"                 # Output folder path
MERGED_FILE="data/law_${DOMAIN}/merged_file.jsonl"          # Path to the merged file
SAMPLE_OUTPUT="data/law_${DOMAIN}/source_docs.jsonl"        # Source pool D with data_id
SAMPLE_SIZE=1000                                            # Unused: all chunks are kept
PREFIX="${DOMAIN}_"                                         # Data ID prefix
MAX_WORKERS=160                                             # Maximum number of worker threads
BATCH_SIZE=500                                              # Batch size
WINDOW_SIZE=8192                                            # Source document length n
TOKENIZER_PATH="llm/Meta-Llama-3-8B"

# Run the Python script
python score/src/0_pre-process.py \
    --tokenizer_path "$TOKENIZER_PATH" \
    --input_folder "$INPUT_FOLDER" \
    --output_folder "$OUTPUT_FOLDER" \
    --max_workers "$MAX_WORKERS" \
    --batch_size "$BATCH_SIZE" \
    --window_size "$WINDOW_SIZE" \
    --merged_file "$MERGED_FILE" \
    --sample_output "$SAMPLE_OUTPUT" \
    --sample_size "$SAMPLE_SIZE" \
    --prefix "$PREFIX"

if [ $? -eq 0 ]; then
    echo "Data processing completed successfully!"
else
    echo "Data processing failed, please check the error messages."
fi
