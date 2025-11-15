"""Script to build retrieval indexes."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_processed_dataset
from src.data.corpus import build_corpus
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.faiss_dense import DenseRetriever
from src.config import PROCESSED_DATA_DIR, INDEX_DIR


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indexes")
    parser.add_argument("--dataset", type=str, required=True, choices=["squad", "triviaqa", "nq"])
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"])
    parser.add_argument("--retrievers", type=str, default="bm25,faiss", help="Comma-separated list: bm25,faiss")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for dense encoding")
    
    args = parser.parse_args()
    
    # Load dataset
    data_path = PROCESSED_DATA_DIR / f"{args.dataset}_{args.split}.json"
    if not data_path.exists():
        print(f"Error: Dataset file not found: {data_path}")
        print("Please run download_data.py first")
        return
    
    print(f"Loading dataset from {data_path}")
    examples = load_processed_dataset(data_path)
    print(f"Loaded {len(examples)} examples")
    
    # Build corpus
    print("\nBuilding passage corpus...")
    passages = build_corpus(args.dataset, examples)
    print(f"Created {len(passages)} passages")
    
    # Build indexes
    retrievers = args.retrievers.split(",")
    
    if "bm25" in retrievers:
        print("\nBuilding BM25 index...")
        bm25_retriever = BM25Retriever(passages)
        bm25_path = INDEX_DIR / f"{args.dataset}_{args.split}_bm25.pkl"
        bm25_retriever.save(bm25_path)
        print(f"Saved BM25 index to {bm25_path}")
    
    if "faiss" in retrievers:
        print("\nBuilding FAISS index...")
        dense_retriever = DenseRetriever(passages, use_gpu=args.use_gpu)
        faiss_path = INDEX_DIR / f"{args.dataset}_{args.split}_faiss.pkl"
        dense_retriever.save(faiss_path)
        print(f"Saved FAISS index to {faiss_path}")
    
    # Save passage metadata
    passages_path = INDEX_DIR / f"{args.dataset}_{args.split}_passages.json"
    with open(passages_path, 'w') as f:
        json.dump(passages, f, indent=2)
    print(f"\nSaved passage metadata to {passages_path}")


if __name__ == "__main__":
    main()

