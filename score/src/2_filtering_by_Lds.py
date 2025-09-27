import argparse
from DataProcess import DateSorted

def main():

    # Arguments for SlideWindow
    parser = argparse.ArgumentParser(description="Sort data with LDS using DateSorted class")
    parser.add_argument('--inference_path', type=str, required=True, help='Path to the inference data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output data')
    parser.add_argument('--file_path', type=str, required=True, help='File path to the input file')

    args = parser.parse_args()

    DateSorted(
        inference_path=args.inference_path,
        file_path=args.file_path,
        output_path=args.output_path,
    ).write_to_file()

if __name__ == '__main__':
    main()
