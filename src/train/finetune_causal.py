"""Fine-tuning script for causal LLMs with QLoRA."""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from datasets import Dataset
from typing import List, Dict, Optional

from ..config import (
    DEFAULT_LORA_R,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_MAX_PROMPT_LENGTH,
    DEFAULT_MAX_ANSWER_LENGTH
)
from ..model.causal_generator import CausalGenerator


def prepare_causal_dataset(
    examples: List[Dict],
    passages_list: Optional[List[List[Dict]]] = None,
    generator: Optional[CausalGenerator] = None,
    max_prompt_length: int = DEFAULT_MAX_PROMPT_LENGTH,
    max_answer_length: int = DEFAULT_MAX_ANSWER_LENGTH
) -> Dataset:
    """Prepare dataset for causal model fine-tuning.
    
    Args:
        examples: List of QA examples with 'question' and 'answers'
        passages_list: Optional list of retrieved passages per example
        generator: CausalGenerator instance for formatting
        max_prompt_length: Maximum prompt length
        max_answer_length: Maximum answer length
        
    Returns:
        HuggingFace Dataset ready for training
    """
    if generator is None:
        generator = CausalGenerator()
    
    if passages_list is None:
        passages_list = [None] * len(examples)
    
    texts = []
    
    for ex, passages in zip(examples, passages_list):
        # Format prompt (without answer)
        prompt = generator.format_prompt(
            ex["question"],
            passages,
            max_prompt_length
        )
        
        # Get answer
        answer = ex["answers"][0] if ex["answers"] else ""
        
        # Full text: prompt + answer
        full_text = prompt + " " + answer
        
        texts.append(full_text)
    
    # Tokenize
    tokenizer = generator.tokenizer
    
    # Tokenize all texts
    tokenized = tokenizer(
        texts,
        max_length=max_prompt_length + max_answer_length,
        truncation=True,
        padding=False
    )
    
    # Create labels (same as input_ids, but -100 for prompt tokens)
    labels = []
    for i, text in enumerate(texts):
        # Tokenize prompt and answer separately to find boundary
        prompt_tokens = tokenizer.encode(
            generator.format_prompt(
                examples[i]["question"],
                passages_list[i],
                max_prompt_length
            ),
            add_special_tokens=False
        )
        
        input_ids = tokenized["input_ids"][i]
        label = [-100] * len(input_ids)
        
        # Only compute loss on answer tokens
        # Find where prompt ends (approximate)
        prompt_len = len(prompt_tokens)
        if prompt_len < len(input_ids):
            label[prompt_len:] = input_ids[prompt_len:]
        
        labels.append(label)
    
    tokenized["labels"] = labels
    
    return Dataset.from_dict(tokenized)


def train_causal_with_qlora(
    model_name: str,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    output_dir: str = "./checkpoints/causal_qlora",
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    epochs: int = 3,
    gradient_accumulation_steps: int = 2,
    fp16: bool = True,
    gradient_checkpointing: bool = True,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 100
):
    """Train causal model with QLoRA.
    
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
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    # Determine target modules based on model architecture
    if "mistral" in model_name.lower():
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    elif "llama" in model_name.lower():
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    else:
        # Default for most models
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
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
        report_to="none",
        optim="paged_adamw_8bit"  # Memory-efficient optimizer
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
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

