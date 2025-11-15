# Retrieval-Augmented QA System

A complete implementation of a retrieval-augmented generation (RAG) system for open-domain question answering, comparing encoder-decoder (T5) and decoder-only (causal) language models.

## Overview

This system implements:
- **Multiple retrieval methods**: BM25 (lexical), FAISS (dense), Hybrid (BM25 + Dense), and optional cross-encoder reranking
- **Two model architectures**: 
  - T5 (encoder-decoder) with LoRA fine-tuning
  - Causal LLMs (decoder-only) with QLoRA fine-tuning
- **Three datasets**: SQuAD v1.1, TriviaQA, and Natural Questions
- **Comprehensive evaluation**: Comparison of retrieval vs. no-retrieval baselines

## Project Structure

```
rag_qa/
├── src/
│   ├── data/           # Dataset loading and corpus building
│   ├── retrieval/      # BM25, FAISS, Hybrid, Reranker
│   ├── model/          # T5 and Causal generators
│   ├── train/          # Fine-tuning scripts
│   └── eval/           # Evaluation metrics and pipeline
├── scripts/            # Main execution scripts
├── configs/            # Configuration files
├── data/               # Raw and processed datasets
├── checkpoints/        # Model checkpoints
└── results/           # Evaluation results
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (for BM25 tokenization):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Quick Start

### 1. Download Datasets

```bash
python scripts/download_data.py --dataset squad --split train
python scripts/download_data.py --dataset squad --split validation
python scripts/download_data.py --dataset triviaqa --split train
python scripts/download_data.py --dataset triviaqa --split validation
```

### 2. Build Retrieval Indexes

```bash
python scripts/build_indexes.py --dataset squad --split train --retrievers bm25,faiss --use_gpu
```

### 3. Train Models

**T5 with RAG (Hybrid retrieval, k=5):**
```bash
python scripts/run_train.py \
    --dataset squad \
    --model_type t5 \
    --retriever hybrid \
    --k 5 \
    --lora \
    --batch_size 8 \
    --epochs 3
```

**Causal LLM with RAG (QLoRA):**
```bash
python scripts/run_train.py \
    --dataset squad \
    --model_type causal \
    --retriever hybrid \
    --k 5 \
    --qlora \
    --batch_size 4 \
    --epochs 3
```

### 4. Evaluate

```bash
python scripts/run_eval.py \
    --dataset squad \
    --model_type t5 \
    --checkpoint checkpoints/squad_t5_hybrid_k5 \
    --settings no_retrieval,bm25,faiss,hybrid \
    --k 1,5,10 \
    --rerank
```

## Design Decisions and Rationale

### Model Selection

**T5 (Flan-T5-Large)**:
- Encoder-decoder architecture excels at conditional generation tasks
- Pre-trained on instruction-following data (Flan), making it well-suited for QA
- LoRA fine-tuning enables efficient adaptation while maintaining performance
- **Rationale**: T5's bidirectional encoder can better understand context-passage relationships compared to autoregressive-only models

**Causal LLM (Mistral-7B-Instruct)**:
- Represents modern production LLMs (decoder-only)
- Demonstrates how RAG works with standard chat models
- QLoRA enables fine-tuning on 16-24GB GPUs
- **Rationale**: Most deployed systems use decoder-only models; important to validate RAG effectiveness with this architecture

### Retrieval Methods

**BM25 (Lexical)**:
- Fast, interpretable, no GPU required
- Excellent for exact keyword matches
- **Rationale**: Strong baseline; often competitive for factoid QA

**Dense Retrieval (BGE-small-en-v1.5)**:
- Captures semantic similarity beyond keywords
- Handles paraphrasing and conceptual queries
- **Rationale**: Modern dense retrievers are state-of-the-art; BGE-small balances quality and speed

**Hybrid (BM25 + Dense)**:
- Combines strengths of both methods
- Min-max normalization + weighted fusion
- **Rationale**: Hybrid retrieval consistently outperforms single methods in practice

**Cross-Encoder Reranker**:
- Re-ranks top-50 candidates from hybrid retrieval
- Computationally expensive but high-quality
- **Rationale**: Reranking provides significant gains with minimal additional latency (only on top candidates)

### Training Strategy

**LoRA/QLoRA**:
- Parameter-efficient fine-tuning reduces memory requirements
- Enables fine-tuning on consumer GPUs
- **Rationale**: Full fine-tuning is expensive; LoRA achieves similar performance with 10-100x fewer parameters

**Input Formatting**:
- **T5**: Simple concatenation `"question: {q} context: <p1> ... <pk>"`
- **Causal**: Chat template with system instruction and structured context
- **Rationale**: T5 benefits from explicit formatting; causal models need instruction-following prompts

**Label Masking (Causal)**:
- Only answer tokens contribute to loss
- Prevents model from learning to copy context verbatim
- **Rationale**: Critical for preventing overfitting to retrieved passages

### Evaluation Metrics

- **Exact Match (EM)**: Strict correctness measure
- **Token-level F1**: Handles partial matches and paraphrasing
- **Recall@K**: Measures retrieval quality (whether answer appears in retrieved passages)
- **Latency**: Practical deployment consideration

## Results Structure

Results are saved in `results/` directory:
- Individual JSON files per setting: `{dataset}_{model_type}_{retriever}_k{k}.json`
- Aggregated CSV: `{dataset}_{model_type}_results.csv`

## Example Results Analysis

After running evaluations, compare:
1. **No retrieval vs. RAG**: Does retrieval improve performance?
2. **Retrieval methods**: BM25 vs. Dense vs. Hybrid
3. **Model types**: T5 vs. Causal LLM
4. **K values**: How many passages are optimal?
5. **Reranking**: Does reranking help?

### Sample SQuAD Results (Flan-T5-Large)

Using `results/squad_t5_results.csv` (SQuAD v1.1 dev, k=20 passages, no reranker):

| Setting | Retriever                  | EM (%) | F1 (%) | Recall@20 (%) | Avg Latency (s) |
|--------|----------------------------|--------|--------|---------------|-----------------|
| faiss  | Dense (FAISS)              | 2.3    | 7.7    | 24.2          | 0.61            |
| hybrid | Hybrid (BM25 + Dense)      | 1.8    | 6.4    | 19.1          | 0.56            |
| hybrid (1 epoch tuned) | Hybrid (BM25 + Dense) | 4.0    | 6.4    | 12.3          | 0.30            |

These preliminary numbers illustrate how dense and hybrid retrieval trade off EM/F1, recall, and latency; you can plug in your own runs by updating the CSVs in `results/`.

## Troubleshooting

**Out of Memory**:
- Use QLoRA for causal models (`--qlora`)
- Reduce batch size
- Use gradient checkpointing (enabled by default)

**Slow Retrieval**:
- Use GPU for dense encoding (`--use_gpu` in build_indexes.py)
- Reduce k value
- Skip reranking for faster evaluation

**Dataset Loading Issues**:
- Some datasets require manual download from HuggingFace
- Check `data/raw/` directory for downloaded files

## Future Improvements
- [ ] Trainable retriever (end-to-end fine-tuning)


