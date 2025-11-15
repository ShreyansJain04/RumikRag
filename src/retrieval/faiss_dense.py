"""Dense retrieval using FAISS and sentence transformers."""
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..config import INDEX_DIR, DEFAULT_EMBEDDING_MODEL


class DenseRetriever:
    """Dense retriever using FAISS and sentence transformers."""
    
    def __init__(
        self,
        passages: Optional[List[Dict]] = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        use_gpu: bool = True
    ):
        """Initialize dense retriever.
        
        Args:
            passages: List of passages with 'text' field. If None, will load from disk.
            model_name: Name of sentence transformer model
            use_gpu: Whether to use GPU for encoding
        """
        self.passages = passages
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        if use_gpu:
            self.encoder = self.encoder.cuda()
        
        self.index = None
        self.passage_embeddings = None
        
        if passages is not None:
            self._build_index(passages)
    
    def _build_index(self, passages: List[Dict], batch_size: int = 32):
        """Build FAISS index from passages.
        
        Args:
            passages: List of passages
            batch_size: Batch size for encoding
        """
        self.passages = passages
        passage_texts = [p["text"] for p in passages]
        
        # Encode passages
        print(f"Encoding {len(passage_texts)} passages...")
        embeddings = []
        for i in tqdm(range(0, len(passage_texts), batch_size)):
            batch = passage_texts[i:i + batch_size]
            batch_embeddings = self.encoder.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalize for cosine similarity
            )
            embeddings.append(batch_embeddings)
        
        self.passage_embeddings = np.vstack(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = self.passage_embeddings.shape[1]
        
        # Use HNSW for fast approximate search
        # M=32 is a good balance between speed and accuracy
        self.index = faiss.IndexHNSWFlat(dimension, 32)
        
        # Add vectors to index
        self.index.add(self.passage_embeddings)
        
        print(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k passages for a query.
        
        Args:
            query: Query string
            top_k: Number of passages to retrieve
            
        Returns:
            List of passages with 'doc_id', 'text', 'score' sorted by score
        """
        if self.index is None:
            raise ValueError("FAISS index not built. Call _build_index() first.")
        
        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Return passages with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.passages):
                passage = self.passages[idx].copy()
                passage["score"] = float(score)
                results.append(passage)
        
        return results
    
    def save(self, filepath: Path):
        """Save FAISS index to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(filepath) + ".faiss")
        
        # Save metadata
        with open(filepath, 'wb') as f:
            pickle.dump({
                'passages': self.passages,
                'model_name': self.model_name,
                'passage_embeddings_shape': self.passage_embeddings.shape if self.passage_embeddings is not None else None
            }, f)
    
    @classmethod
    def load(cls, filepath: Path, use_gpu: bool = True):
        """Load FAISS index from disk."""
        # Load metadata
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Load FAISS index
        index = faiss.read_index(str(filepath) + ".faiss")
        
        # Initialize encoder without rebuilding the index; reuse precomputed FAISS index
        retriever = cls(
            passages=None,
            model_name=data['model_name'],
            use_gpu=use_gpu
        )
        retriever.passages = data['passages']
        retriever.index = index
        
        return retriever

