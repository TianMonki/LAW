import argparse
from DataRank import DateSorted

def main():
    parser = argparse.ArgumentParser(description="Sort and filter data with LDS using Rank Aggregation")
    
    # Changed to accept one or more inference paths
    parser.add_argument('--inference_paths', type=str, nargs='+', required=True, 
                        help='One or more paths to the inference data files.')
    
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Path to save the final filtered output data.')
    
    parser.add_argument('--file_path', type=str, required=True, 
                        help='File path to the original unfiltered input file.')

    args = parser.parse_args()

    # Pass the list of paths to the DateSorted class
    sorter = DateSorted(
        inference_paths=args.inference_paths,
        file_path=args.file_path,
        output_path=args.output_path,
    )
    sorter.write_to_file()

if __name__ == '__main__':
    main()