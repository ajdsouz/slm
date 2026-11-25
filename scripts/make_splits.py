import argparse
from datasets import load_from_disk

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path-to-dataset",
        type=str,
        help="Path to unsplit dataset"
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        help="Test split ratio"
    )

    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save new dataset at"
    )

    args = parser.parse_args()

    dataset = load_from_disk(args.path_to_dataset)
    splitted_dataset = dataset['train'].train_test_split(test_size=args.test_ratio)
    splitted_dataset = splitted_dataset.flatten_indices()
    splitted_dataset.save_to_disk(args.save_path)

if __name__ == "__main__":
    main()
