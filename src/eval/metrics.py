"""Evaluation metrics for QA."""
import re
from typing import List, Dict, Set
from collections import Counter


def normalize_answer(text: str) -> str:
    """Normalize answer text for evaluation.
    
    Args:
        text: Answer text
        
    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Compute exact match.
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        True if exact match after normalization
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score.
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 1.0 if len(pred_tokens) == len(truth_tokens) else 0.0
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def compute_metrics(
    predictions: List[str],
    references: List[List[str]],
    metrics: List[str] = ["exact_match", "f1"]
) -> Dict[str, float]:
    """Compute evaluation metrics.
    
    Args:
        predictions: List of predicted answers
        references: List of lists of reference answers (one per prediction)
        metrics: List of metric names to compute
        
    Returns:
        Dictionary of metric scores
    """
    results = {}
    
    if "exact_match" in metrics:
        em_scores = []
        for pred, refs in zip(predictions, references):
            # EM is 1 if any reference matches
            em = max([exact_match(pred, ref) for ref in refs])
            em_scores.append(em)
        results["exact_match"] = sum(em_scores) / len(em_scores) if em_scores else 0.0
    
    if "f1" in metrics:
        f1_scores = []
        for pred, refs in zip(predictions, references):
            # F1 is max over all references
            f1 = max([f1_score(pred, ref) for ref in refs])
            f1_scores.append(f1)
        results["f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    return results


def recall_at_k(
    retrieved_passages: List[Dict],
    ground_truth_answers: List[str],
    k: int = 5
) -> float:
    """Compute recall@k: whether any ground truth answer appears in retrieved passages.
    
    Args:
        retrieved_passages: List of retrieved passages (with 'text' field)
        ground_truth_answers: List of ground truth answer strings
        k: Number of top passages to consider
        
    Returns:
        Recall@k score (0.0 or 1.0)
    """
    if not retrieved_passages or not ground_truth_answers:
        return 0.0
    
    # Check top-k passages
    top_k_passages = retrieved_passages[:k]
    passage_texts = [p.get("text", "").lower() for p in top_k_passages]
    combined_text = " ".join(passage_texts)
    
    # Normalize ground truth answers
    normalized_answers = [normalize_answer(ans) for ans in ground_truth_answers]
    normalized_text = normalize_answer(combined_text)
    
    # Check if any answer appears in passages
    for ans in normalized_answers:
        if ans and ans in normalized_text:
            return 1.0
    
    return 0.0


def compute_retrieval_metrics(
    retrieved_passages_list: List[List[Dict]],
    ground_truth_answers_list: List[List[str]],
    k: int = 5
) -> Dict[str, float]:
    """Compute retrieval metrics across a dataset.
    
    Args:
        retrieved_passages_list: List of retrieved passage lists (one per question)
        ground_truth_answers_list: List of ground truth answer lists (one per question)
        k: Number of top passages to consider
        
    Returns:
        Dictionary of retrieval metrics
    """
    recall_scores = []
    
    for retrieved, answers in zip(retrieved_passages_list, ground_truth_answers_list):
        recall = recall_at_k(retrieved, answers, k)
        recall_scores.append(recall)
    
    return {
        f"recall_at_{k}": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    }

