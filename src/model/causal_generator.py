"""Decoder-only LLM generator for causal QA."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
from tqdm import tqdm

from ..config import DEFAULT_CAUSAL_MODEL, DEFAULT_MAX_PROMPT_LENGTH, DEFAULT_MAX_ANSWER_LENGTH


class CausalGenerator:
    """Decoder-only LLM generator for question answering."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_CAUSAL_MODEL,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """Initialize causal generator.
        
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
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except:
                # Extract base model name from adapter config if available
                try:
                    import json
                    adapter_config_path = os.path.join(model_name, "adapter_config.json")
                    if os.path.exists(adapter_config_path):
                        with open(adapter_config_path, 'r') as f:
                            config = json.load(f)
                            base_model = config.get("base_model_name_or_path", DEFAULT_CAUSAL_MODEL)
                            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                    else:
                        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CAUSAL_MODEL)
                except:
                    self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CAUSAL_MODEL)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if is_checkpoint:
            # Load base model first
            base_model_name = DEFAULT_CAUSAL_MODEL
            try:
                import json
                adapter_config_path = os.path.join(model_name, "adapter_config.json")
                if os.path.exists(adapter_config_path):
                    with open(adapter_config_path, 'r') as f:
                        config = json.load(f)
                        base_model_name = config.get("base_model_name_or_path", DEFAULT_CAUSAL_MODEL)
            except:
                pass
            
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            # Load PEFT adapter
            self.model = PeftModel.from_pretrained(base_model, model_name)
            self.model = self.model.merge_and_unload()  # Merge adapter weights
        elif load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        elif load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            )
            self.model = self.model.to(self.device)
        
        self.model.eval()
    
    def format_prompt(
        self,
        question: str,
        passages: Optional[List[Dict]] = None,
        max_prompt_length: int = DEFAULT_MAX_PROMPT_LENGTH
    ) -> str:
        """Format prompt for causal model.
        
        Args:
            question: Question string
            passages: Optional list of retrieved passages
            max_prompt_length: Maximum prompt length
            
        Returns:
            Formatted prompt string
        """
        # System instruction
        system_msg = "Use the provided context to answer the question. If the answer cannot be found in the context, say 'I don't know'."
        
        if passages is None or len(passages) == 0:
            # No retrieval case
            user_msg = f"Q: {question}"
        else:
            # Format passages
            context_parts = []
            for i, passage in enumerate(passages, 1):
                text = passage.get("text", "")
                context_parts.append(f"[{i}] {text}")
            
            context = "\n".join(context_parts)
            user_msg = f"Context:\n{context}\n\nQ: {question}"
        
        # Format according to model's chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback formatting for models without chat template
            prompt = f"{system_msg}\n\n{user_msg}\n\nA:"
        
        # Truncate if too long
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > max_prompt_length:
            # Truncate context part
            question_part = f"{system_msg}\n\nQ: {question}\n\nA:"
            question_tokens = self.tokenizer.encode(question_part)
            max_context_length = max_prompt_length - len(question_tokens) - 20  # Buffer
            
            if passages:
                context_tokens = self.tokenizer.encode("\n".join(context_parts))
                if len(context_tokens) > max_context_length:
                    context_tokens = context_tokens[:max_context_length]
                    context = self.tokenizer.decode(context_tokens, skip_special_tokens=True)
                    prompt = f"{system_msg}\n\nContext:\n{context}\n\nQ: {question}\n\nA:"
        
        return prompt
    
    def generate(
        self,
        question: str,
        passages: Optional[List[Dict]] = None,
        max_new_tokens: int = DEFAULT_MAX_ANSWER_LENGTH,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """Generate answer for a question.
        
        Args:
            question: Question string
            passages: Optional list of retrieved passages
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation kwargs
            
        Returns:
            Generated answer string
        """
        # Format prompt
        prompt = self.format_prompt(question, passages)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=DEFAULT_MAX_PROMPT_LENGTH
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode only the generated part
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up answer (remove common prefixes)
        answer = answer.strip()
        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()
        
        return answer
    
    def generate_batch(
        self,
        questions: List[str],
        passages_list: Optional[List[List[Dict]]] = None,
        batch_size: int = 4,  # Smaller batch for causal models
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
        
        answers = []
        
        for i in tqdm(range(0, len(questions), batch_size), desc="Generating"):
            batch_questions = questions[i:i + batch_size]
            batch_passages = passages_list[i:i + batch_size]
            
            batch_answers = []
            for q, p in zip(batch_questions, batch_passages):
                answer = self.generate(q, p, **generation_kwargs)
                batch_answers.append(answer)
            
            answers.extend(batch_answers)
        
        return answers

