"""Corpus building and chunking utilities."""
import re
from typing import Dict, List, Tuple
from tqdm import tqdm
import tiktoken

from ..config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_STRIDE


def tokenize_text(text: str, encoding_name: str = "cl100k_base") -> List[int]:
    """Tokenize text using tiktoken."""
    enc = tiktoken.get_encoding(encoding_name)
    return enc.encode(text)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text."""
    return len(tokenize_text(text, encoding_name))


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stride: int = DEFAULT_CHUNK_STRIDE,
    tokenizer_name: str = "cl100k_base"
) -> List[str]:
    """Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Target number of tokens per chunk
        stride: Number of tokens to overlap between chunks
        tokenizer_name: Tokenizer encoding name
        
    Returns:
        List of text chunks
    """
    # Simple sentence-aware chunking
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sent_tokens = count_tokens(sentence, tokenizer_name)
        
        if current_tokens + sent_tokens > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap
            if stride > 0:
                # Keep last few sentences for overlap
                overlap_tokens = 0
                overlap_sents = []
                for sent in reversed(current_chunk):
                    sent_toks = count_tokens(sent, tokenizer_name)
                    if overlap_tokens + sent_toks <= stride:
                        overlap_sents.insert(0, sent)
                        overlap_tokens += sent_toks
                    else:
                        break
                current_chunk = overlap_sents
                current_tokens = overlap_tokens
            else:
                current_chunk = []
                current_tokens = 0
        
        current_chunk.append(sentence)
        current_tokens += sent_tokens
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Fallback: if no chunks created, use simple token-based splitting
    if not chunks:
        enc = tiktoken.get_encoding(tokenizer_name)
        tokens = tokenize_text(text, tokenizer_name)
        for i in range(0, len(tokens), chunk_size - stride):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)
    
    return chunks


def build_corpus_from_squad(examples: List[Dict]) -> List[Dict]:
    """Build passage corpus from SQuAD examples.
    
    Args:
        examples: List of SQuAD examples with 'context' field
        
    Returns:
        List of passages with keys: doc_id, title, text, source_id
    """
    passages = []
    doc_id = 0
    
    seen_contexts = set()
    
    for ex in tqdm(examples, desc="Building corpus from SQuAD"):
        context = ex.get("context", "")
        if not context:
            continue
        
        # Deduplicate contexts
        context_hash = hash(context.lower().strip())
        if context_hash in seen_contexts:
            continue
        seen_contexts.add(context_hash)
        
        # Chunk the context
        chunks = chunk_text(context)
        
        for chunk_idx, chunk in enumerate(chunks):
            passages.append({
                "doc_id": f"squad_{doc_id}_{chunk_idx}",
                "title": ex.get("title", ""),
                "text": chunk,
                "source_id": ex["id"]
            })
        
        doc_id += 1
    
    return passages


def build_corpus_from_triviaqa(examples: List[Dict]) -> List[Dict]:
    """Build passage corpus from TriviaQA examples.
    
    Args:
        examples: List of TriviaQA examples with 'evidence' field
        
    Returns:
        List of passages with keys: doc_id, title, text, source_id
    """
    passages = []
    doc_id = 0
    
    for ex in tqdm(examples, desc="Building corpus from TriviaQA"):
        evidence_list = ex.get("evidence", [])
        if not evidence_list:
            # Use context if available
            evidence_list = [ex.get("context", "")]
        
        for evidence in evidence_list:
            if not evidence:
                continue
            
            # Chunk each evidence paragraph
            chunks = chunk_text(evidence)
            
            for chunk_idx, chunk in enumerate(chunks):
                passages.append({
                    "doc_id": f"triviaqa_{doc_id}_{chunk_idx}",
                    "title": ex.get("title", ""),
                    "text": chunk,
                    "source_id": ex["id"]
                })
        
        doc_id += 1
    
    return passages


def build_corpus_from_nq(examples: List[Dict]) -> List[Dict]:
    """Build passage corpus from Natural Questions examples.
    
    Args:
        examples: List of NQ examples
        
    Returns:
        List of passages with keys: doc_id, title, text, source_id
    """
    passages = []
    doc_id = 0
    
    seen_contexts = set()
    
    for ex in tqdm(examples, desc="Building corpus from NQ"):
        context = ex.get("context", "")
        if not context:
            continue
        
        # Simple HTML stripping (basic)
        context = re.sub(r'<[^>]+>', '', context)
        context = ' '.join(context.split())
        
        context_hash = hash(context.lower().strip())
        if context_hash in seen_contexts:
            continue
        seen_contexts.add(context_hash)
        
        # Chunk the context
        chunks = chunk_text(context)
        
        for chunk_idx, chunk in enumerate(chunks):
            passages.append({
                "doc_id": f"nq_{doc_id}_{chunk_idx}",
                "title": ex.get("title", ""),
                "text": chunk,
                "source_id": ex["id"]
            })
        
        doc_id += 1
    
    return passages


def build_corpus(dataset_name: str, examples: List[Dict]) -> List[Dict]:
    """Build passage corpus from dataset examples.
    
    Args:
        dataset_name: 'squad', 'triviaqa', or 'nq'
        examples: List of dataset examples
        
    Returns:
        List of passages
    """
    if dataset_name.lower() == "squad":
        return build_corpus_from_squad(examples)
    elif dataset_name.lower() == "triviaqa":
        return build_corpus_from_triviaqa(examples)
    elif dataset_name.lower() == "nq" or dataset_name.lower() == "natural_questions":
        return build_corpus_from_nq(examples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_passage_to_qa_mapping(
    examples: List[Dict],
    passages: List[Dict]
) -> Dict[str, List[str]]:
    """Create mapping from passage doc_id to question IDs that reference it.
    
    Args:
        examples: List of QA examples
        passages: List of passages
        
    Returns:
        Dict mapping doc_id to list of question IDs
    """
    mapping = {}
    
    # Build reverse mapping from source_id to question IDs
    source_to_questions = {}
    for ex in examples:
        source_id = ex["id"]
        if source_id not in source_to_questions:
            source_to_questions[source_id] = []
        source_to_questions[source_id].append(ex["id"])
    
    # Map passages to questions
    for passage in passages:
        source_id = passage.get("source_id")
        if source_id in source_to_questions:
            mapping[passage["doc_id"]] = source_to_questions[source_id]
        else:
            mapping[passage["doc_id"]] = []
    
    return mapping

