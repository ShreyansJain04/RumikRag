"""Run comprehensive evaluation of all trained models."""
import subprocess
import sys
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
RESULTS_DIR = Path(__file__).parent.parent / "results"

def run_evaluation(dataset, model_type, checkpoint, settings, k_values):
    """Run evaluation for a specific checkpoint."""
    cmd = [
        "python", "scripts/run_eval.py",
        "--dataset", dataset,
        "--model_type", model_type,
        "--checkpoint", str(checkpoint),
        "--settings", settings,
        "--k", k_values
    ]
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint.name}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error evaluating {checkpoint.name}:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True

def main():
    dataset = "squad"
    
    # Evaluate T5 models
    t5_checkpoints = [
        CHECKPOINT_DIR / "squad_t5_none_k5",
        CHECKPOINT_DIR / "squad_t5_bm25_k5",
        CHECKPOINT_DIR / "squad_t5_hybrid_k5"
    ]
    
    for checkpoint in t5_checkpoints:
        if checkpoint.exists():
            # Evaluate with same retrieval method as training
            if "none" in checkpoint.name:
                settings = "no_retrieval"
            elif "bm25" in checkpoint.name:
                settings = "no_retrieval,bm25"
            elif "hybrid" in checkpoint.name:
                settings = "no_retrieval,bm25,faiss,hybrid"
            else:
                settings = "no_retrieval,bm25,faiss,hybrid"
            
            run_evaluation(dataset, "t5", checkpoint, settings, "1,5,10")
        else:
            print(f"Checkpoint not found: {checkpoint}")
    
    # Evaluate zero-shot baseline
    print(f"\n{'='*60}")
    print("Evaluating zero-shot baseline")
    print(f"{'='*60}")
    cmd = [
        "python", "scripts/run_eval.py",
        "--dataset", dataset,
        "--model_type", "t5",
        "--model_name", "google/flan-t5-base",
        "--settings", "no_retrieval,bm25,faiss,hybrid",
        "--k", "1,5,10"
    ]
    subprocess.run(cmd, cwd=Path(__file__).parent.parent)

if __name__ == "__main__":
    main()


