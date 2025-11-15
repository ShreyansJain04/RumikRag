"""BM25 retrieval implementation."""
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from ..config import INDEX_DIR


def tokenize(text: str) -> List[str]:
    """Tokenize text for BM25."""
    tokens = word_tokenize(text.lower())
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    return tokens


class BM25Retriever:
    """BM25-based retriever."""
    
    def __init__(self, passages: Optional[List[Dict]] = None):
        """Initialize BM25 retriever.
        
        Args:
            passages: List of passages with 'text' field. If None, will load from disk.
        """
        self.passages = passages
        self.bm25 = None
        self.passage_texts = []
        
        if passages is not None:
            self._build_index(passages)
    
    def _build_index(self, passages: List[Dict]):
        """Build BM25 index from passages."""
        self.passages = passages
        self.passage_texts = [p["text"] for p in passages]
        
        # Tokenize all passages
        tokenized_passages = [tokenize(text) for text in self.passage_texts]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_passages)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k passages for a query.
        
        Args:
            query: Query string
            top_k: Number of passages to retrieve
            
        Returns:
            List of passages with 'doc_id', 'text', 'score' sorted by score
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call _build_index() first.")
        
        # Tokenize query
        tokenized_query = tokenize(query)
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Return passages with scores
        results = []
        for idx in top_indices:
            passage = self.passages[idx].copy()
            passage["score"] = float(scores[idx])
            results.append(passage)
        
        return results
    
    def save(self, filepath: Path):
        """Save BM25 index to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'passages': self.passages,
                'bm25': self.bm25,
                'passage_texts': self.passage_texts
            }, f)
    
    @classmethod
    def load(cls, filepath: Path):
        """Load BM25 index from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        retriever = cls()
        retriever.passages = data['passages']
        retriever.bm25 = data['bm25']
        retriever.passage_texts = data['passage_texts']
        
        return retriever

