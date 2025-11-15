"""Script to download and prepare datasets."""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import (
    load_squad,
    load_triviaqa,
    load_natural_questions,
    save_processed_dataset
)
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Download and prepare datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["squad", "triviaqa", "nq", "all"])
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on number of examples to keep (random subsample with fixed seed)",
    )

    args = parser.parse_args()

    datasets_to_process = []
    if args.dataset == "all":
        datasets_to_process = ["squad", "triviaqa", "nq"]
    else:
        datasets_to_process = [args.dataset]

    for dataset_name in datasets_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name.upper()} ({args.split})")
        print(f"{'='*60}")

        try:
            if dataset_name == "squad":
                examples = load_squad(split=args.split)
            elif dataset_name == "triviaqa":
                examples = load_triviaqa(split=args.split)
            elif dataset_name == "nq":
                examples = load_natural_questions(split=args.split)

            original_len = len(examples)
            print(f"Loaded {original_len} examples")

            # Optional random subsample to reduce dataset size
            if args.limit is not None:
                if args.limit <= 0:
                    print(f"Requested limit {args.limit} is non-positive; keeping all {original_len} examples.")
                elif args.limit >= original_len:
                    print(f"Requested limit {args.limit} >= dataset size {original_len}; keeping all examples.")
                else:
                    random.seed(42)
                    examples = random.sample(examples, args.limit)
                    print(f"Subsampled to {len(examples)} examples (from {original_len})")

            # Save processed dataset
            output_path = PROCESSED_DATA_DIR / f"{dataset_name}_{args.split}.json"
            save_processed_dataset(examples, output_path)
            print(f"Saved to {output_path}")

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

