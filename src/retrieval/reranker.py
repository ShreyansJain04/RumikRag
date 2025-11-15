"""Cross-encoder reranker for retrieved passages."""
from typing import List, Dict
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from ..config import DEFAULT_RERANKER_MODEL, DEFAULT_RERANK_CANDIDATES, DEFAULT_RERANK_TOP_K


class Reranker:
    """Cross-encoder reranker."""
    
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        """Initialize reranker.
        
        Args:
            model_name: Name of cross-encoder model
        """
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        passages: List[Dict],
        top_k: int = DEFAULT_RERANK_TOP_K
    ) -> List[Dict]:
        """Rerank passages using cross-encoder.
        
        Args:
            query: Query string
            passages: List of passages to rerank (with 'text' field)
            top_k: Number of top passages to return
            
        Returns:
            Reranked list of passages with updated 'score' field
        """
        if not passages:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, p["text"]] for p in passages]
        
        # Get scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Update scores and sort
        reranked = []
        for passage, score in zip(passages, scores):
            passage_copy = passage.copy()
            passage_copy["score"] = float(score)
            reranked.append(passage_copy)
        
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]
    
    def rerank_batch(
        self,
        queries: List[str],
        passages_list: List[List[Dict]],
        top_k: int = DEFAULT_RERANK_TOP_K,
        batch_size: int = 32
    ) -> List[List[Dict]]:
        """Rerank passages for multiple queries in batch.
        
        Args:
            queries: List of query strings
            passages_list: List of lists of passages (one per query)
            top_k: Number of top passages to return per query
            batch_size: Batch size for cross-encoder
            
        Returns:
            List of reranked passage lists
        """
        results = []
        
        for query, passages in tqdm(
            zip(queries, passages_list),
            total=len(queries),
            desc="Reranking"
        ):
            results.append(self.rerank(query, passages, top_k))
        
        return results

