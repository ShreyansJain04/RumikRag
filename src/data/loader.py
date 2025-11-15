"""Dataset loading utilities for SQuAD, TriviaQA, and Natural Questions."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm

from ..config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_squad(split: str = "validation") -> List[Dict]:
    """Load SQuAD v1.1 dataset.
    
    Args:
        split: 'train' or 'validation'
        
    Returns:
        List of examples with keys: id, question, answers, context, title
    """
    dataset = load_dataset("squad", split=split)
    examples = []
    
    for item in dataset:
        examples.append({
            "id": item["id"],
            "question": item["question"],
            "answers": item["answers"]["text"],  # List of answer strings
            "context": item["context"],
            "title": item["title"]
        })
    
    return examples


def load_triviaqa(split: str = "validation") -> List[Dict]:
    """Load TriviaQA dataset (Wikipedia evidence).
    
    Args:
        split: 'train' or 'validation'
        
    Returns:
        List of examples with keys: id, question, answers, evidence
    """
    # TriviaQA has different splits - use the Wikipedia evidence version
    try:
        dataset = load_dataset("trivia_qa", "rc.wikipedia", split=split)
    except:
        # Fallback to rc.nocontext if wikipedia not available
        dataset = load_dataset("trivia_qa", "rc.nocontext", split=split)
    
    examples = []
    for item in dataset:
        # TriviaQA has answer aliases
        answer_data = item.get("answer", {})
        if isinstance(answer_data, dict):
            answers = answer_data.get("aliases", [])
            if not answers:
                answers = [answer_data.get("value", "")]
        else:
            answers = [str(answer_data)] if answer_data else [""]
        
        # Get evidence paragraphs
        evidence = []
        if "search_results" in item:
            search_results = item["search_results"]
            if isinstance(search_results, list):
                for result in search_results:
                    if isinstance(result, dict) and "search_context" in result:
                        evidence.append(result["search_context"])
                    elif isinstance(result, str):
                        # Sometimes search_results is a list of strings
                        evidence.append(result)
        
        # Get title
        title = ""
        if "entity_pages" in item and item["entity_pages"]:
            entity_pages = item["entity_pages"]
            if isinstance(entity_pages, list) and len(entity_pages) > 0:
                if isinstance(entity_pages[0], dict):
                    title = entity_pages[0].get("title", "")
        
        examples.append({
            "id": item.get("question_id", str(item.get("id", ""))),
            "question": item.get("question", ""),
            "answers": answers,
            "evidence": evidence,  # List of evidence paragraphs
            "title": title
        })
    
    return examples


def load_natural_questions(split: str = "validation") -> List[Dict]:
    """Load Natural Questions dataset.
    
    Args:
        split: 'train' or 'validation'
        
    Returns:
        List of examples with keys: id, question, answers, context, title
    """
    try:
        # Try to load the simplified version
        dataset = load_dataset("natural_questions", split=split)
    except Exception as e:
        print(f"Warning: Natural Questions loading failed: {e}")
        return []
    
    examples = []
    for item in dataset:
        # Get ID - try different possible field names
        example_id = item.get("id") or item.get("example_id") or str(item.get("idx", ""))
        
        # Extract question
        question_text = ""
        if "question" in item:
            if isinstance(item["question"], dict):
                question_text = item["question"].get("text", "")
            elif isinstance(item["question"], str):
                question_text = item["question"]
        elif "question_text" in item:
            question_text = item["question_text"]
        
        # Extract answers
        answers = []
        if "short_answers" in item and item["short_answers"]:
            for ans in item["short_answers"]:
                if isinstance(ans, dict) and "text" in ans:
                    answers.append(ans["text"])
                elif isinstance(ans, str):
                    answers.append(ans)
        
        # Also check annotations
        if not answers and "annotations" in item:
            annotations = item["annotations"]
            if isinstance(annotations, list) and len(annotations) > 0:
                ann = annotations[0]
                if "short_answers" in ann:
                    for ans in ann["short_answers"]:
                        if isinstance(ans, dict):
                            # Extract text from spans
                            if "text" in ans:
                                answers.append(ans["text"])
        
        # Get context from document
        context = ""
        title = ""
        if "document" in item:
            doc = item["document"]
            if isinstance(doc, dict):
                # Extract title
                title = doc.get("title", "")
                
                # Extract text from HTML or plain text
                if "html" in doc:
                    import re
                    # Simple HTML stripping
                    html_text = doc["html"]
                    context = re.sub(r'<[^>]+>', '', html_text)
                    context = ' '.join(context.split())
                elif "text" in doc:
                    context = doc["text"]
                elif "plain_text" in doc:
                    context = doc["plain_text"]
        
        # If no answers found, use empty string
        if not answers:
            answers = [""]
        
        examples.append({
            "id": example_id,
            "question": question_text,
            "answers": answers,
            "context": context,
            "title": title
        })
    
    return examples


def normalize_answer(text: str) -> str:
    """Normalize answer text for evaluation."""
    import re
    # Lowercase, remove punctuation, normalize whitespace
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text


def prepare_qa_examples(dataset_name: str, split: str = "validation") -> List[Dict]:
    """Load and prepare QA examples from a dataset.
    
    Args:
        dataset_name: 'squad', 'triviaqa', or 'nq'
        split: 'train' or 'validation'
        
    Returns:
        List of standardized examples with keys: id, question, answers, context, title
    """
    if dataset_name.lower() == "squad":
        examples = load_squad(split)
    elif dataset_name.lower() == "triviaqa":
        examples = load_triviaqa(split)
    elif dataset_name.lower() == "nq" or dataset_name.lower() == "natural_questions":
        examples = load_natural_questions(split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Standardize format
    standardized = []
    for ex in examples:
        # Ensure answers is a list
        answers = ex.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        if not answers:
            answers = [""]
        
        standardized.append({
            "id": ex["id"],
            "question": ex["question"],
            "answers": answers,
            "context": ex.get("context", ""),
            "title": ex.get("title", ""),
            "evidence": ex.get("evidence", [])  # For TriviaQA
        })
    
    return standardized


def save_processed_dataset(examples: List[Dict], output_path: Path):
    """Save processed dataset to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)


def load_processed_dataset(input_path: Path) -> List[Dict]:
    """Load processed dataset from JSON."""
    with open(input_path, 'r') as f:
        return json.load(f)

