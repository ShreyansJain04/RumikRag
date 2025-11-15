"""Configuration management for RAG QA system."""
import os
from pathlib import Path
from typing import Optional

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEX_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations
DEFAULT_T5_MODEL = "google/flan-t5-large"
DEFAULT_CAUSAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Training defaults
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_EPOCHS = 3
DEFAULT_MAX_SOURCE_LENGTH = 1024  # For T5
DEFAULT_MAX_TARGET_LENGTH = 32  # For T5
DEFAULT_MAX_PROMPT_LENGTH = 1536  # For causal models
DEFAULT_MAX_ANSWER_LENGTH = 64  # For causal models

# Retrieval defaults
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_STRIDE = 50
DEFAULT_TOP_K = 5
DEFAULT_RERANK_TOP_K = 10
DEFAULT_RERANK_CANDIDATES = 50

# LoRA defaults
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.1

# Random seed
RANDOM_SEED = 42

