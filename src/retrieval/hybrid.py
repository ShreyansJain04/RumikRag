"""Hybrid retrieval combining BM25 and dense retrieval."""
from typing import List, Dict
import numpy as np

from .bm25 import BM25Retriever
from .faiss_dense import DenseRetriever


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] using min-max normalization."""
    if not scores:
        return scores
    
    scores_array = np.array(scores)
    min_score = scores_array.min()
    max_score = scores_array.max()
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    normalized = (scores_array - min_score) / (max_score - min_score)
    return normalized.tolist()


class HybridRetriever:
    """Hybrid retriever combining BM25 and dense retrieval."""
    
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        alpha: float = 0.5
    ):
        """Initialize hybrid retriever.
        
        Args:
            bm25_retriever: BM25 retriever instance
            dense_retriever: Dense retriever instance
            alpha: Weight for dense retrieval (1-alpha for BM25)
        """
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.alpha = alpha
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k passages using hybrid scoring.
        
        Args:
            query: Query string
            top_k: Number of passages to retrieve
            
        Returns:
            List of passages with 'doc_id', 'text', 'score' sorted by score
        """
        # Retrieve from both retrievers (get more candidates for fusion)
        candidate_k = min(top_k * 3, 100)  # Get 3x candidates for fusion
        
        bm25_results = self.bm25_retriever.retrieve(query, top_k=candidate_k)
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k)
        
        # Create passage ID to score mapping
        passage_scores = {}
        
        # Extract BM25 scores
        bm25_scores = [r["score"] for r in bm25_results]
        bm25_scores_norm = normalize_scores(bm25_scores)
        
        for result, score in zip(bm25_results, bm25_scores_norm):
            doc_id = result["doc_id"]
            passage_scores[doc_id] = {
                "passage": result,
                "bm25_score": score,
                "dense_score": 0.0
            }
        
        # Extract dense scores
        dense_scores = [r["score"] for r in dense_results]
        dense_scores_norm = normalize_scores(dense_scores)
        
        for result, score in zip(dense_results, dense_scores_norm):
            doc_id = result["doc_id"]
            if doc_id not in passage_scores:
                passage_scores[doc_id] = {
                    "passage": result,
                    "bm25_score": 0.0,
                    "dense_score": score
                }
            else:
                passage_scores[doc_id]["dense_score"] = score
        
        # Compute hybrid scores
        hybrid_results = []
        for doc_id, data in passage_scores.items():
            hybrid_score = (
                (1 - self.alpha) * data["bm25_score"] +
                self.alpha * data["dense_score"]
            )
            passage = data["passage"].copy()
            passage["score"] = hybrid_score
            passage["bm25_score"] = data["bm25_score"]
            passage["dense_score"] = data["dense_score"]
            hybrid_results.append(passage)
        
        # Sort by hybrid score and return top-k
        hybrid_results.sort(key=lambda x: x["score"], reverse=True)
        return hybrid_results[:top_k]

