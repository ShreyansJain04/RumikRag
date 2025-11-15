"""Evaluation pipeline for RAG QA system."""
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from tqdm import tqdm

from ..model.generator import T5Generator
from ..model.causal_generator import CausalGenerator
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.faiss_dense import DenseRetriever
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.reranker import Reranker
from .metrics import compute_metrics, compute_retrieval_metrics
from ..config import RESULTS_DIR


class RAGEvaluator:
    """Evaluator for RAG QA system."""
    
    def __init__(
        self,
        model_type: str = "t5",
        model_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        reranker: Optional[Reranker] = None
    ):
        """Initialize evaluator.
        
        Args:
            model_type: 't5' or 'causal'
            model_name: HuggingFace model name
            checkpoint_path: Path to fine-tuned checkpoint
            bm25_retriever: BM25 retriever instance
            dense_retriever: Dense retriever instance
            reranker: Optional reranker instance
        """
        self.model_type = model_type
        
        # Load generator
        if model_type == "t5":
            if checkpoint_path:
                self.generator = T5Generator(model_name=checkpoint_path)
            else:
                self.generator = T5Generator(model_name=model_name or "google/flan-t5-large")
        else:
            if checkpoint_path:
                self.generator = CausalGenerator(model_name=checkpoint_path)
            else:
                self.generator = CausalGenerator(model_name=model_name or "mistralai/Mistral-7B-Instruct-v0.2")
        
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.reranker = reranker
    
    def evaluate_no_retrieval(
        self,
        examples: List[Dict],
        batch_size: int = 16
    ) -> Dict:
        """Evaluate without retrieval (baseline).
        
        Args:
            examples: List of QA examples
            batch_size: Batch size for generation
            
        Returns:
            Dictionary of metrics and results
        """
        questions = [ex["question"] for ex in examples]
        references = [ex["answers"] for ex in examples]
        
        # Generate answers
        start_time = time.time()
        predictions = self.generator.generate_batch(questions, batch_size=batch_size)
        elapsed_time = time.time() - start_time
        
        # Compute metrics
        metrics = compute_metrics(predictions, references)
        metrics["avg_latency"] = elapsed_time / len(examples)
        
        return {
            "metrics": metrics,
            "predictions": predictions,
            "references": references
        }
    
    def evaluate_with_retrieval(
        self,
        examples: List[Dict],
        retriever_type: str = "bm25",
        top_k: int = 5,
        use_reranker: bool = False,
        batch_size: int = 16
    ) -> Dict:
        """Evaluate with retrieval.
        
        Args:
            examples: List of QA examples
            retriever_type: 'bm25', 'dense', or 'hybrid'
            top_k: Number of passages to retrieve
            use_reranker: Whether to use reranker
            batch_size: Batch size for generation
            
        Returns:
            Dictionary of metrics and results
        """
        questions = [ex["question"] for ex in examples]
        references = [ex["answers"] for ex in examples]
        
        # Retrieve passages
        retrieved_passages_list = []
        for question in tqdm(questions, desc="Retrieving"):
            if retriever_type == "bm25":
                if self.bm25_retriever is None:
                    raise ValueError("BM25 retriever not initialized")
                passages = self.bm25_retriever.retrieve(question, top_k=top_k)
            elif retriever_type == "dense" or retriever_type == "faiss":
                if self.dense_retriever is None:
                    raise ValueError("Dense retriever not initialized")
                passages = self.dense_retriever.retrieve(question, top_k=top_k)
            elif retriever_type == "hybrid":
                if self.bm25_retriever is None or self.dense_retriever is None:
                    raise ValueError("Both retrievers must be initialized for hybrid")
                hybrid = HybridRetriever(self.bm25_retriever, self.dense_retriever)
                passages = hybrid.retrieve(question, top_k=top_k)
            else:
                raise ValueError(f"Unknown retriever type: {retriever_type}")
            
            # Rerank if requested
            if use_reranker and self.reranker is not None:
                passages = self.reranker.rerank(question, passages)
            
            retrieved_passages_list.append(passages)
        
        # Generate answers
        start_time = time.time()
        predictions = self.generator.generate_batch(
            questions,
            passages_list=retrieved_passages_list,
            batch_size=batch_size
        )
        elapsed_time = time.time() - start_time
        
        # Compute metrics
        metrics = compute_metrics(predictions, references)
        retrieval_metrics = compute_retrieval_metrics(
            retrieved_passages_list,
            references,
            k=top_k
        )
        metrics.update(retrieval_metrics)
        metrics["avg_latency"] = elapsed_time / len(examples)
        
        return {
            "metrics": metrics,
            "predictions": predictions,
            "references": references,
            "retrieved_passages": retrieved_passages_list
        }
    
    def evaluate(
        self,
        examples: List[Dict],
        setting: str = "no_retrieval",
        retriever_type: str = "bm25",
        top_k: int = 5,
        use_reranker: bool = False,
        batch_size: int = 16
    ) -> Dict:
        """Evaluate with specified setting.
        
        Args:
            examples: List of QA examples
            setting: 'no_retrieval' or 'with_retrieval'
            retriever_type: 'bm25', 'dense', or 'hybrid' (only for with_retrieval)
            top_k: Number of passages to retrieve
            use_reranker: Whether to use reranker
            batch_size: Batch size for generation
            
        Returns:
            Dictionary of metrics and results
        """
        if setting == "no_retrieval":
            return self.evaluate_no_retrieval(examples, batch_size=batch_size)
        else:
            return self.evaluate_with_retrieval(
                examples,
                retriever_type=retriever_type,
                top_k=top_k,
                use_reranker=use_reranker,
                batch_size=batch_size
            )


def save_results(results: Dict, output_path: Path):
    """Save evaluation results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    serializable = {}
    for key, value in results.items():
        if key == "metrics":
            serializable[key] = value
        elif key in ["predictions", "references"]:
            serializable[key] = value
        elif key == "retrieved_passages":
            # Simplify passages for JSON
            serializable[key] = [
                [{"doc_id": p["doc_id"], "text": p["text"][:200]} for p in passages]
                for passages in value
            ]
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)


def aggregate_results(results_list: List[Dict], output_path: Path):
    """Aggregate multiple evaluation results into a CSV.
    
    Args:
        results_list: List of result dictionaries, each with 'setting', 'metrics', etc.
        output_path: Path to save CSV
    """
    rows = []
    for result in results_list:
        row = {
            "setting": result.get("setting", ""),
            "model_type": result.get("model_type", ""),
            "retriever": result.get("retriever", ""),
            "top_k": result.get("top_k", ""),
            "reranker": result.get("reranker", False),
            **result.get("metrics", {})
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

