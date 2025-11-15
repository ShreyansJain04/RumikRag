"""Fine-tuning script for T5 models with LoRA."""
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from typing import List, Dict, Optional

from ..config import (
    DEFAULT_LORA_R,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_MAX_SOURCE_LENGTH,
    DEFAULT_MAX_TARGET_LENGTH
)
from ..model.generator import T5Generator


def prepare_t5_dataset(
    examples: List[Dict],
    passages_list: Optional[List[List[Dict]]] = None,
    generator: Optional[T5Generator] = None,
    max_source_length: int = DEFAULT_MAX_SOURCE_LENGTH,
    max_target_length: int = DEFAULT_MAX_TARGET_LENGTH
) -> Dataset:
    """Prepare dataset for T5 fine-tuning.
    
    Args:
        examples: List of QA examples with 'question' and 'answers'
        passages_list: Optional list of retrieved passages per example
        generator: T5Generator instance for formatting
        max_source_length: Maximum source length
        max_target_length: Maximum target length
        
    Returns:
        HuggingFace Dataset ready for training
    """
    if generator is None:
        generator = T5Generator()
    
    if passages_list is None:
        passages_list = [None] * len(examples)
    
    inputs = []
    targets = []
    
    for ex, passages in zip(examples, passages_list):
        # Format input
        input_text = generator.format_input(
            ex["question"],
            passages,
            max_source_length
        )
        
        # Use first answer as target
        target_text = ex["answers"][0] if ex["answers"] else ""
        
        inputs.append(input_text)
        targets.append(target_text)
    
    # Tokenize
    tokenizer = generator.tokenizer
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        truncation=True,
        padding=True
    )
    
    labels = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True,
        padding=True
    )
    
    # Replace padding token id with -100 for loss computation
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    
    return Dataset.from_dict(model_inputs)


def train_t5_with_lora(
    model_name: str,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    output_dir: str = "./checkpoints/t5_lora",
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    learning_rate: float = 2e-4,
    batch_size: int = 8,
    epochs: int = 3,
    gradient_accumulation_steps: int = 1,
    fp16: bool = True,
    gradient_checkpointing: bool = True,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 100
):
    """Train T5 model with LoRA.
    
    Args:
        model_name: HuggingFace model name
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        output_dir: Output directory for checkpoints
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        learning_rate: Learning rate
        batch_size: Batch size
        epochs: Number of epochs
        gradient_accumulation_steps: Gradient accumulation steps
        fp16: Whether to use FP16
        gradient_checkpointing: Whether to use gradient checkpointing
        save_steps: Steps between saves
        eval_steps: Steps between evaluations
        logging_steps: Steps between logging
    """
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q", "v"],  # T5 attention modules
        lora_dropout=lora_dropout,
        bias="none"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing (PEFT-friendly) and disable cache
    if gradient_checkpointing:
        # Standard HF gradient checkpointing switch
        model.gradient_checkpointing_enable()
        # Required for some PEFT adapters so inputs participate in the graph
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        # Avoid HF warning and ensure compatibility
        if hasattr(model, "config"):
            model.config.use_cache = False
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        report_to="none"
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

