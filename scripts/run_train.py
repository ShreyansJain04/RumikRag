"""Script to train models with/without retrieval."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_processed_dataset
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.faiss_dense import DenseRetriever
from src.train.finetune import prepare_t5_dataset, train_t5_with_lora
from src.train.finetune_causal import prepare_causal_dataset, train_causal_with_qlora
from tqdm import tqdm
from src.model.generator import T5Generator
from src.model.causal_generator import CausalGenerator
from src.config import (
    PROCESSED_DATA_DIR,
    INDEX_DIR,
    CHECKPOINT_DIR,
    DEFAULT_T5_MODEL,
    DEFAULT_CAUSAL_MODEL
)


def main():
    parser = argparse.ArgumentParser(description="Train models with/without retrieval")
    parser.add_argument("--dataset", type=str, required=True, choices=["squad", "triviaqa", "nq"])
    parser.add_argument("--model_type", type=str, required=True, choices=["t5", "causal"])
    parser.add_argument("--model_name", type=str, help="Override default model name")
    parser.add_argument("--retriever", type=str, choices=["bm25", "faiss", "hybrid", "none"], default="none")
    parser.add_argument("--k", type=int, default=5, help="Number of passages to retrieve")
    parser.add_argument("--rerank", action="store_true", help="Use reranker")
    parser.add_argument("--lora", action="store_true", help="Use LoRA for T5")
    parser.add_argument("--qlora", action="store_true", help="Use QLoRA for causal models")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, help="Output directory for checkpoint")
    # Mixed precision control
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        help="Use FP16 mixed precision training (default: enabled)",
    )
    parser.add_argument(
        "--no_fp16",
        dest="fp16",
        action="store_false",
        help="Disable FP16 mixed precision and train in full precision/bf16",
    )
    parser.set_defaults(fp16=True)
    
    args = parser.parse_args()
    
    # Load training data
    train_path = PROCESSED_DATA_DIR / f"{args.dataset}_train.json"
    if not train_path.exists():
        print(f"Error: Training data not found: {train_path}")
        print("Please run download_data.py first")
        return
    
    print(f"Loading training data from {train_path}")
    train_examples = load_processed_dataset(train_path)
    print(f"Loaded {len(train_examples)} training examples")
    
    # Load validation data if available
    val_path = PROCESSED_DATA_DIR / f"{args.dataset}_validation.json"
    val_examples = None
    if val_path.exists():
        val_examples = load_processed_dataset(val_path)
        print(f"Loaded {len(val_examples)} validation examples")
    
    # Prepare retrieval if needed
    passages_list = None
    if args.retriever != "none":
        # Load corpus
        passages_path = INDEX_DIR / f"{args.dataset}_train_passages.json"
        if not passages_path.exists():
            print(f"Error: Passage corpus not found: {passages_path}")
            print("Please run build_indexes.py first")
            return
        
        with open(passages_path, 'r') as f:
            passages = json.load(f)
        
        # Load retrievers
        bm25_retriever = None
        dense_retriever = None
        
        if args.retriever in ["bm25", "hybrid"]:
            bm25_path = INDEX_DIR / f"{args.dataset}_train_bm25.pkl"
            if bm25_path.exists():
                bm25_retriever = BM25Retriever.load(bm25_path)
            else:
                print(f"Warning: BM25 index not found: {bm25_path}")
        
        if args.retriever in ["faiss", "hybrid"]:
            faiss_path = INDEX_DIR / f"{args.dataset}_train_faiss.pkl"
            if faiss_path.exists():
                dense_retriever = DenseRetriever.load(faiss_path)
            else:
                print(f"Warning: FAISS index not found: {faiss_path}")
        
        # Retrieve passages for training examples
        print(f"\nRetrieving passages using {args.retriever}...")
        passages_list = []
        
        if args.retriever == "bm25":
            if bm25_retriever is None:
                raise ValueError("BM25 retriever not available")
            for ex in tqdm(train_examples, desc="Retrieving (bm25)"):
                retrieved = bm25_retriever.retrieve(ex["question"], top_k=args.k)
                passages_list.append(retrieved)
        elif args.retriever == "faiss":
            if dense_retriever is None:
                raise ValueError("Dense retriever not available")
            for ex in tqdm(train_examples, desc="Retrieving (faiss)"):
                retrieved = dense_retriever.retrieve(ex["question"], top_k=args.k)
                passages_list.append(retrieved)
        elif args.retriever == "hybrid":
            from src.retrieval.hybrid import HybridRetriever
            if bm25_retriever is None or dense_retriever is None:
                raise ValueError("Both retrievers must be available for hybrid")
            hybrid = HybridRetriever(bm25_retriever, dense_retriever)
            for ex in tqdm(train_examples, desc="Retrieving (hybrid)"):
                retrieved = hybrid.retrieve(ex["question"], top_k=args.k)
                passages_list.append(retrieved)
        
        print(f"Retrieved passages for {len(passages_list)} examples")
    
    # Prepare dataset
    print("\nPreparing dataset...")
    if args.model_type == "t5":
        model_name = args.model_name or DEFAULT_T5_MODEL
        generator = T5Generator(model_name=model_name)
        train_dataset = prepare_t5_dataset(train_examples, passages_list, generator)
        val_dataset = None
        if val_examples:
            val_passages_list = None  # Could retrieve for val too, but skip for now
            val_dataset = prepare_t5_dataset(val_examples, val_passages_list, generator)
    else:
        model_name = args.model_name or DEFAULT_CAUSAL_MODEL
        generator = CausalGenerator(model_name=model_name, load_in_4bit=args.qlora)
        train_dataset = prepare_causal_dataset(train_examples, passages_list, generator)
        val_dataset = None
        if val_examples:
            val_passages_list = None
            val_dataset = prepare_causal_dataset(val_examples, val_passages_list, generator)
    
    print(f"Prepared dataset with {len(train_dataset)} examples")
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        setting_name = f"{args.retriever}_k{args.k}"
        if args.rerank:
            setting_name += "_rerank"
        output_dir = CHECKPOINT_DIR / f"{args.dataset}_{args.model_type}_{setting_name}"
    
    # Train
    print(f"\nTraining {args.model_type} model...")
    print(f"Output directory: {output_dir}")
    
    if args.model_type == "t5":
        if not args.lora:
            print("Warning: Training T5 without LoRA may require significant memory")
        train_t5_with_lora(
            model_name=model_name,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            output_dir=str(output_dir),
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=args.fp16,
            gradient_checkpointing=False
        )
    else:
        if not args.qlora:
            print("Warning: Training causal model without QLoRA may require significant memory")
            print("Consider using --qlora flag")
        train_causal_with_qlora(
            model_name=model_name,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            output_dir=str(output_dir),
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=args.fp16,
            gradient_checkpointing=True
        )
    
    print(f"\nTraining complete! Checkpoint saved to {output_dir}")


if __name__ == "__main__":
    main()

