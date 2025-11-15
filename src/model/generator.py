"""T5-based generator for seq2seq QA."""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Optional
from tqdm import tqdm

from ..config import DEFAULT_T5_MODEL, DEFAULT_MAX_SOURCE_LENGTH, DEFAULT_MAX_TARGET_LENGTH


class T5Generator:
    """T5-based generator for question answering."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_T5_MODEL,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """Initialize T5 generator.
        
        Args:
            model_name: HuggingFace model name or path to checkpoint
            device: Device to use ('cuda' or 'cpu')
            load_in_8bit: Whether to load model in 8-bit
            load_in_4bit: Whether to load model in 4-bit
        """
        self.model_name = model_name
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Check if this is a checkpoint path (contains PEFT adapter)
        from peft import PeftModel
        import os
        
        is_checkpoint = os.path.exists(model_name) and os.path.isdir(model_name)
        
        # Load tokenizer
        if is_checkpoint:
            # Try to load tokenizer from checkpoint, fallback to model name
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            except:
                # Extract base model name from adapter config if available
                try:
                    import json
                    adapter_config_path = os.path.join(model_name, "adapter_config.json")
                    if os.path.exists(adapter_config_path):
                        with open(adapter_config_path, 'r') as f:
                            config = json.load(f)
                            base_model = config.get("base_model_name_or_path", DEFAULT_T5_MODEL)
                            self.tokenizer = T5Tokenizer.from_pretrained(base_model)
                    else:
                        self.tokenizer = T5Tokenizer.from_pretrained(DEFAULT_T5_MODEL)
                except:
                    self.tokenizer = T5Tokenizer.from_pretrained(DEFAULT_T5_MODEL)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Load model
        if is_checkpoint:
            # Load base model first
            base_model_name = DEFAULT_T5_MODEL
            try:
                import json
                adapter_config_path = os.path.join(model_name, "adapter_config.json")
                if os.path.exists(adapter_config_path):
                    with open(adapter_config_path, 'r') as f:
                        config = json.load(f)
                        base_model_name = config.get("base_model_name_or_path", DEFAULT_T5_MODEL)
            except:
                pass
            
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                base_model = T5ForConditionalGeneration.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
                base_model = base_model.to(self.device)
            
            # Load PEFT adapter
            self.model = PeftModel.from_pretrained(base_model, model_name)
            self.model = self.model.merge_and_unload()  # Merge adapter weights
        elif load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        elif load_in_8bit:
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model = self.model.to(self.device)
        
        self.model.eval()
    
    def format_input(
        self,
        question: str,
        passages: Optional[List[Dict]] = None,
        max_source_length: int = DEFAULT_MAX_SOURCE_LENGTH
    ) -> str:
        """Format input for T5.
        
        Args:
            question: Question string
            passages: Optional list of retrieved passages
            max_source_length: Maximum source length
            
        Returns:
            Formatted input string
        """
        if passages is None or len(passages) == 0:
            return f"question: {question}"
        
        # Format passages
        context_parts = []
        for i, passage in enumerate(passages, 1):
            text = passage.get("text", "")
            context_parts.append(f"<p{i}> {text} </p{i}>")
        
        context = " ".join(context_parts)
        
        # Truncate if too long
        input_text = f"question: {question} context: {context}"
        tokens = self.tokenizer.encode(input_text)
        
        if len(tokens) > max_source_length:
            # Truncate context
            question_tokens = self.tokenizer.encode(f"question: {question}")
            max_context_length = max_source_length - len(question_tokens) - 10  # Buffer
            
            # Rebuild with truncated context
            context_tokens = self.tokenizer.encode(context)
            if len(context_tokens) > max_context_length:
                context_tokens = context_tokens[:max_context_length]
                context = self.tokenizer.decode(context_tokens, skip_special_tokens=True)
            
            input_text = f"question: {question} context: {context}"
        
        return input_text
    
    def generate(
        self,
        question: str,
        passages: Optional[List[Dict]] = None,
        max_length: int = DEFAULT_MAX_TARGET_LENGTH,
        num_beams: int = 4,
        temperature: float = 0.7,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """Generate answer for a question.
        
        Args:
            question: Question string
            passages: Optional list of retrieved passages
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation kwargs
            
        Returns:
            Generated answer string
        """
        # Format input
        input_text = self.format_input(question, passages)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=DEFAULT_MAX_SOURCE_LENGTH
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                **kwargs
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def generate_batch(
        self,
        questions: List[str],
        passages_list: Optional[List[List[Dict]]] = None,
        batch_size: int = 8,
        **generation_kwargs
    ) -> List[str]:
        """Generate answers for a batch of questions.
        
        Args:
            questions: List of question strings
            passages_list: Optional list of passage lists (one per question)
            batch_size: Batch size for generation
            **generation_kwargs: Generation kwargs
            
        Returns:
            List of generated answer strings
        """
        if passages_list is None:
            passages_list = [None] * len(questions)

        answers: List[str] = []

        for i in tqdm(range(0, len(questions), batch_size), desc="Generating"):
            batch_questions = questions[i : i + batch_size]
            batch_passages = passages_list[i : i + batch_size]

            # Format inputs for the whole batch
            batch_inputs_text = [
                self.format_input(q, p) for q, p in zip(batch_questions, batch_passages)
            ]

            # Tokenize as a single padded batch
            inputs = self.tokenizer(
                batch_inputs_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=DEFAULT_MAX_SOURCE_LENGTH,
            ).to(self.device)

            # Batched generation on GPU
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=DEFAULT_MAX_TARGET_LENGTH,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=False,
                    **generation_kwargs,
                )

            # Decode all answers in the batch
            batch_answers = [
                self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs
            ]
            answers.extend(batch_answers)

        return answers

