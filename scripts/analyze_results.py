"""Analysis utilities for inspecting RAG QA results in detail.

This script lets you:
- Summarize metrics across result files (T5 vs causal, retrieval vs no-retrieval)
- Inspect specific questions across multiple methods:
  - Question text and gold answers
  - Retrieved contexts (BM25 / FAISS / hybrid)
  - Generated answers from each model / setting, with per-example EM and F1

Usage examples (from repo root):

  # Show a quick summary of all SQuAD experiments
  python scripts/analyze_results.py --dataset squad --summary

  # Inspect the first SQuAD example across all available methods
  python scripts/analyze_results.py --dataset squad --index 0 --num_examples 1

  # Inspect a specific SQuAD id
  python scripts/analyze_results.py --dataset squad --id 573786b51c4567190057448d
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

# Add repo root to path (same pattern as run_eval.py)
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR  # type: ignore
from src.eval.metrics import exact_match, f1_score  # type: ignore


@dataclass
class Example:
    idx: int
    id: str
    question: str
    answers: List[str]
    context: str
    title: str


@dataclass
class ExperimentMeta:
    dataset: str
    model_type: str
    setting: str
    top_k: Optional[int]
    rerank: bool
    path: Path


def load_dataset(dataset: str) -> List[Example]:
    """Load processed dataset as a list of Examples."""
    path = PROCESSED_DATA_DIR / f"{dataset}_validation.json"
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    with path.open("r") as f:
        raw = json.load(f)

    examples: List[Example] = []
    for idx, ex in enumerate(raw):
        examples.append(
            Example(
                idx=idx,
                id=ex.get("id", str(idx)),
                question=ex.get("question", ""),
                answers=ex.get("answers", []),
                context=ex.get("context", ""),
                title=ex.get("title", ""),
            )
        )
    return examples


def parse_experiment_name(path: Path) -> Optional[ExperimentMeta]:
    """Parse result filename into metadata.

    Expected patterns from run_eval.py:
      {dataset}_{model_type}_{setting}_k{k}.json
      {dataset}_{model_type}_{setting}_k{k}_rerank.json
      {dataset}_{model_type}_{setting}_k1.json (for no_retrieval variants)
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None

    dataset = parts[0]
    model_type = parts[1]
    setting = parts[2]
    top_k: Optional[int] = None
    rerank = False

    for p in parts[3:]:
        if p.startswith("k") and len(p) > 1 and p[1:].isdigit():
            try:
                top_k = int(p[1:])
            except ValueError:
                top_k = None
        elif p == "rerank":
            rerank = True

    return ExperimentMeta(
        dataset=dataset,
        model_type=model_type,
        setting=setting,
        top_k=top_k,
        rerank=rerank,
        path=path,
    )


def discover_experiments(dataset: str) -> List[ExperimentMeta]:
    """Find all JSON result files for a given dataset."""
    results: List[ExperimentMeta] = []
    if not RESULTS_DIR.exists():
        return results

    for path in sorted(RESULTS_DIR.glob(f"{dataset}_*.json")):
        meta = parse_experiment_name(path)
        if meta is not None and meta.dataset == dataset:
            results.append(meta)
    return results


def load_results(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def summarize_experiments(dataset: str) -> None:
    """Print a compact summary of all experiments and their metrics."""
    experiments = discover_experiments(dataset)
    if not experiments:
        print(f"No result JSON files found for dataset '{dataset}' in {RESULTS_DIR}")
        return

    print(f"\nAvailable experiments for dataset='{dataset}':")
    print("-" * 80)
    header = (
        f"{'experiment':40s}  {'model':6s}  {'setting':11s}  "
        f"{'k':>3s}  {'rerank':6s}  {'EM':>7s}  {'F1':>7s}  "
        f"{'R@5':>7s}  {'latency':>8s}"
    )
    print(header)
    print("-" * 80)

    for meta in experiments:
        data = load_results(meta.path)
        metrics = data.get("metrics", {})
        em = metrics.get("exact_match", 0.0)
        f1 = metrics.get("f1", 0.0)
        recall_at_5 = metrics.get("recall_at_5")
        latency = metrics.get("avg_latency", 0.0)

        name = meta.path.stem[:40]
        k_str = "" if meta.top_k is None else str(meta.top_k)
        rerank_str = "yes" if meta.rerank else "no"
        r5_str = f"{recall_at_5:.4f}" if recall_at_5 is not None else ""

        print(
            f"{name:40s}  {meta.model_type:6s}  {meta.setting:11s}  "
            f"{k_str:>3s}  {rerank_str:6s}  "
            f"{em:7.4f}  {f1:7.4f}  {r5_str:>7s}  {latency:8.4f}"
        )

    print("-" * 80)


def build_index_by_id(examples: List[Example]) -> Dict[str, int]:
    return {ex.id: ex.idx for ex in examples}


def select_examples(
    examples: List[Example],
    idx: Optional[int],
    num_examples: int,
    example_id: Optional[str],
) -> List[Example]:
    """Select one or more examples by index or id."""
    if example_id is not None:
        by_id = build_index_by_id(examples)
        if example_id not in by_id:
            raise ValueError(f"Example id '{example_id}' not found in dataset")
        start_idx = by_id[example_id]
    else:
        if idx is None:
            start_idx = 0
        else:
            if idx < 0 or idx >= len(examples):
                raise ValueError(
                    f"Index {idx} out of range for dataset of size {len(examples)}"
                )
            start_idx = idx

    end_idx = min(start_idx + num_examples, len(examples))
    return examples[start_idx:end_idx]


def per_example_metrics(
    prediction: str,
    references: List[str],
) -> Tuple[float, float]:
    """Compute EM and F1 for a single prediction against a list of references."""
    if not references:
        return 0.0, 0.0
    em_scores = [1.0 if exact_match(prediction, ref) else 0.0 for ref in references]
    f1_scores = [f1_score(prediction, ref) for ref in references]
    return max(em_scores), max(f1_scores)


def print_single_example(
    example: Example,
    exp_results: Dict[str, Dict],
    exp_meta: Dict[str, ExperimentMeta],
    max_passages: int = 3,
) -> None:
    """Pretty-print one example across multiple experiments."""
    print("\n" + "=" * 80)
    print(f"Example #{example.idx}  id={example.id}  title={example.title}")
    print("-" * 80)
    print(f"Q: {example.question}")
    print(f"Gold answers: {example.answers}")
    print(f"Context (truncated): {example.context[:400]}{'...' if len(example.context) > 400 else ''}")
    print("-" * 80)

    for name, results in exp_results.items():
        meta = exp_meta[name]
        predictions: List[str] = results.get("predictions", [])
        references: List[List[str]] = results.get("references", [])

        if example.idx >= len(predictions):
            print(f"[{name}] WARNING: no prediction for idx={example.idx}")
            continue

        pred = predictions[example.idx]
        refs_for_example: List[str]
        if references and example.idx < len(references):
            refs_for_example = references[example.idx]
        else:
            refs_for_example = example.answers

        em, f1 = per_example_metrics(pred, refs_for_example)

        label = f"{meta.model_type}-{meta.setting}"
        if meta.top_k is not None and meta.setting != "no_retrieval":
            label += f"-k{meta.top_k}"
        if meta.rerank:
            label += "-rerank"

        print(f"[{label}]")
        print(f"  Prediction: {pred}")
        print(f"  EM={em:.1f}  F1={f1:.3f}")

        retrieved_passages = results.get("retrieved_passages")
        if retrieved_passages is not None:
            if example.idx < len(retrieved_passages):
                passages = retrieved_passages[example.idx][:max_passages]
                print(f"  Retrieved passages (top {len(passages)}):")
                for i, p in enumerate(passages, 1):
                    doc_id = p.get("doc_id", "")
                    text = p.get("text", "")
                    snippet = text[:200] + ("..." if len(text) > 200 else "")
                    print(f"    [{i}] {doc_id}: {snippet}")
            else:
                print("  WARNING: no retrieved passages for this index")

        print("-" * 80)


def analyze_examples(
    dataset: str,
    index: Optional[int],
    num_examples: int,
    example_id: Optional[str],
    methods: Optional[List[str]],
) -> None:
    """Inspect one or more examples across multiple experiments."""
    examples = load_dataset(dataset)
    selected = select_examples(examples, index, num_examples, example_id)

    experiments = discover_experiments(dataset)
    if not experiments:
        print(f"No experiments found for dataset='{dataset}' in {RESULTS_DIR}")
        return

    # Filter experiments if specific methods are provided (match on stem)
    if methods:
        method_set = set(methods)
        experiments = [e for e in experiments if e.path.stem in method_set]
        if not experiments:
            print(
                f"No experiments matched the given methods: {sorted(method_set)}. "
                f"Available stems include: {[e.path.stem for e in discover_experiments(dataset)]}"
            )
            return

    # Load all selected experiments into memory once
    exp_results: Dict[str, Dict] = {}
    exp_meta: Dict[str, ExperimentMeta] = {}
    for meta in experiments:
        data = load_results(meta.path)
        exp_results[meta.path.stem] = data
        exp_meta[meta.path.stem] = meta

    print(
        f"\nInspecting {len(selected)} example(s) across {len(exp_results)} experiment(s) "
        f"for dataset='{dataset}'"
    )

    for ex in selected:
        print_single_example(ex, exp_results, exp_meta)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze RAG QA results (metrics + per-question breakdown)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="squad",
        help="Dataset name (e.g., squad, triviaqa, nq).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary table of all experiments for the dataset.",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="0-based index of the first example to inspect.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1,
        help="Number of consecutive examples to inspect starting at --index (or id).",
    )
    parser.add_argument(
        "--id",
        dest="example_id",
        type=str,
        help="Example id to inspect (overrides --index if provided).",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        help=(
            "Optional list of experiment stems to include "
            "(e.g., squad_t5_no_retrieval_k5 squad_t5_hybrid_k5). "
            "Defaults to all experiments for the dataset."
        ),
    )

    args = parser.parse_args()

    if args.summary:
        summarize_experiments(args.dataset)

    if args.index is not None or args.example_id is not None:
        analyze_examples(
            dataset=args.dataset,
            index=args.index,
            num_examples=args.num_examples,
            example_id=args.example_id,
            methods=args.methods,
        )
    elif not args.summary:
        # Default to summary if no specific analysis options are provided
        summarize_experiments(args.dataset)


if __name__ == "__main__":
    main()


