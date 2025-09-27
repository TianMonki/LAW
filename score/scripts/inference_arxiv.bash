#!/bin/bash

# ==============================================================================
#  Configuration
# ==============================================================================
# Set input parameters that are constant for all runs
file_path="data/process_arxiv/sample_output.jsonl"  # Input file path
batch_size=48  # Batch size
error_log="error.log"  # Define error log file

# Check if batch_size is a valid number
if ! [[ "$batch_size" =~ ^[0-9]+$ ]]; then
    echo "Error: Batch size must be a number."
    exit 1
fi

# Clear the existing error log file before starting
> "$error_log"

# ==============================================================================
#  Main Loop
# ==============================================================================
# Loop through the specified margin values
for margin in 2; do
    echo "----------------------------------------------------"
    echo "--- Starting process for margin=$margin ---"
    echo "----------------------------------------------------"

    # --- Step 1: Inference ---
    # Define file paths that depend on the current margin value
    output_file="data/process_arxiv/output_inference_$margin.jsonl"

    echo "Running inference for margin=$margin..."
    echo "Output will be saved to: $output_file"

    # Run the first inference script
    if ! accelerate launch LongAttn/src/1_inference_dp.py "$file_path" "$output_file" "$batch_size" "$margin" 2>>"$error_log"; then
        echo "Inference script failed for margin=$margin. Check $error_log for details."
        exit 1
    fi
    echo "Inference for margin=$margin completed successfully."

    # # --- Step 2: Filtering ---
    # # Define parameters for the second script (DateSorted)
    # inference_path="$output_file"
    # output_path="data/process_code/final_output_$margin.jsonl"

    # echo "Running DateSorted processing for margin=$margin..."
    # echo "Final output will be saved to: $output_path"
    
    # # Run the second script (DateSorted)
    # if ! python LongAttn/src/2_filtering_by_Lds.py \
    #     --inference_path "$inference_path" \
    #     --output_path "$output_path" \
    #     --file_path "$file_path" 2>>"$error_log"; then
    #     echo "DateSorted processing script failed for margin=$margin. Check $error_log for details."
    #     exit 1
    # fi

    # echo "Processing for margin=$margin complete. Results saved to $output_path."
    # echo "" # Add a blank line for better readability
done

echo "----------------------------------------------------"
echo "All processing complete."
echo "----------------------------------------------------"