"""Script to evaluate models with/without retrieval."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_processed_dataset
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.faiss_dense import DenseRetriever
from src.retrieval.reranker import Reranker
from src.eval.evaluate import RAGEvaluator, save_results, aggregate_results
from src.config import (
    PROCESSED_DATA_DIR,
    INDEX_DIR,
    RESULTS_DIR,
    CHECKPOINT_DIR,
    DEFAULT_T5_MODEL,
    DEFAULT_CAUSAL_MODEL
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate models with/without retrieval")
    parser.add_argument("--dataset", type=str, required=True, choices=["squad", "triviaqa", "nq"])
    parser.add_argument("--model_type", type=str, required=True, choices=["t5", "causal"])
    parser.add_argument("--model_name", type=str, help="Override default model name")
    parser.add_argument("--checkpoint", type=str, help="Path to fine-tuned checkpoint")
    parser.add_argument("--settings", type=str, default="no_retrieval", help="Comma-separated: no_retrieval,bm25,faiss,hybrid")
    parser.add_argument("--k", type=str, default="5", help="Comma-separated list of k values")
    parser.add_argument("--rerank", action="store_true", help="Use reranker for retrieval settings")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Limit number of examples to evaluate")
    
    args = parser.parse_args()
    
    # Load evaluation data
    val_path = PROCESSED_DATA_DIR / f"{args.dataset}_validation.json"
    if not val_path.exists():
        print(f"Error: Validation data not found: {val_path}")
        print("Please run download_data.py first")
        return
    
    print(f"Loading evaluation data from {val_path}")
    examples = load_processed_dataset(val_path)
    if args.limit:
        examples = examples[:args.limit]
    print(f"Evaluating on {len(examples)} examples")
    
    # Load retrievers if needed
    settings = args.settings.split(",")
    needs_retrieval = any(s != "no_retrieval" for s in settings)
    
    bm25_retriever = None
    dense_retriever = None
    reranker = None
    
    if needs_retrieval:
        # Load passage corpus
        passages_path = INDEX_DIR / f"{args.dataset}_train_passages.json"
        if not passages_path.exists():
            print(f"Warning: Passage corpus not found: {passages_path}")
            print("Retrieval will use validation split corpus")
            passages_path = INDEX_DIR / f"{args.dataset}_validation_passages.json"
        
        # Load BM25 if needed
        if any(s in ["bm25", "hybrid"] for s in settings):
            bm25_path = INDEX_DIR / f"{args.dataset}_train_bm25.pkl"
            if bm25_path.exists():
                bm25_retriever = BM25Retriever.load(bm25_path)
                print("Loaded BM25 retriever")
            else:
                print(f"Warning: BM25 index not found: {bm25_path}")
        
        # Load FAISS if needed
        if any(s in ["faiss", "hybrid"] for s in settings):
            faiss_path = INDEX_DIR / f"{args.dataset}_train_faiss.pkl"
            if faiss_path.exists():
                dense_retriever = DenseRetriever.load(faiss_path)
                print("Loaded FAISS retriever")
            else:
                print(f"Warning: FAISS index not found: {faiss_path}")
        
        # Load reranker if requested
        if args.rerank:
            reranker = Reranker()
            print("Loaded reranker")
    
    # Determine model name/checkpoint
    model_name = args.model_name
    checkpoint_path = args.checkpoint
    
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            checkpoint_path = None
    
    if not checkpoint_path and not model_name:
        model_name = DEFAULT_T5_MODEL if args.model_type == "t5" else DEFAULT_CAUSAL_MODEL
    
    # Evaluate each setting
    k_values = [int(k) for k in args.k.split(",")]
    all_results = []
    
    for setting in settings:
        for k in k_values:
            print(f"\n{'='*60}")
            print(f"Evaluating: {setting} (k={k})")
            print(f"{'='*60}")
            
            # Create evaluator
            evaluator = RAGEvaluator(
                model_type=args.model_type,
                model_name=model_name,
                checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                bm25_retriever=bm25_retriever,
                dense_retriever=dense_retriever,
                reranker=reranker
            )
            
            # Evaluate
            if setting == "no_retrieval":
                result = evaluator.evaluate_no_retrieval(examples, batch_size=args.batch_size)
            else:
                use_rerank = args.rerank and setting != "no_retrieval"
                result = evaluator.evaluate_with_retrieval(
                    examples,
                    retriever_type=setting,
                    top_k=k,
                    use_reranker=use_rerank,
                    batch_size=args.batch_size
                )
            
            # Print metrics
            print("\nMetrics:")
            for metric, value in result["metrics"].items():
                print(f"  {metric}: {value:.4f}")
            
            # Save individual result
            result_file = RESULTS_DIR / f"{args.dataset}_{args.model_type}_{setting}_k{k}.json"
            if args.rerank and setting != "no_retrieval":
                result_file = RESULTS_DIR / f"{args.dataset}_{args.model_type}_{setting}_k{k}_rerank.json"
            
            save_results(result, result_file)
            
            # Prepare for aggregation
            result["setting"] = setting
            result["model_type"] = args.model_type
            result["retriever"] = setting if setting != "no_retrieval" else "none"
            result["top_k"] = k if setting != "no_retrieval" else None
            result["reranker"] = args.rerank and setting != "no_retrieval"
            all_results.append(result)
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("Aggregating results...")
    print(f"{'='*60}")
    
    agg_file = RESULTS_DIR / f"{args.dataset}_{args.model_type}_results.csv"
    df = aggregate_results(all_results, agg_file)
    print(f"\nAggregated results saved to {agg_file}")
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

