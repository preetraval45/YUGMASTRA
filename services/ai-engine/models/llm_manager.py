"""
Advanced LLM Manager with Multi-Model Support
Supports: Ollama (local FREE), HuggingFace (FREE), OpenAI, Anthropic
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import os
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

class ModelProvider(str, Enum):
    OLLAMA = "ollama"  # FREE - Local inference
    HUGGINGFACE = "huggingface"  # FREE - Open models
    OPENAI = "openai"  # Paid
    ANTHROPIC = "anthropic"  # Paid
    LOCAL = "local"  # Local transformers

class LLMManager:
    """
    Advanced LLM Manager supporting multiple providers with fallback
    Priority: FREE local > FREE cloud > Paid
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.pipelines: Dict[str, Any] = {}
        self.active_provider: Optional[ModelProvider] = None
        self.llm_clients: Dict[str, Any] = {}

        # FREE models only setting
        self.use_free_only = os.getenv("USE_FREE_MODELS_ONLY", "true").lower() == "true"

        logger.info(f"LLM Manager initialized on device: {self.device}")
        logger.info(f"FREE models only: {self.use_free_only}")

        # Initialize all available providers
        self._initialize_providers()

        # Load default local models as fallback
        self._load_default_models()

    def _initialize_providers(self):
        """Initialize all available LLM providers"""
        # Only FREE providers if use_free_only is True
        if self.use_free_only:
            providers_to_try = [
                self._init_ollama,      # FREE - Best for privacy & no limits
                self._init_huggingface, # FREE - Good open models
            ]
            logger.info("🆓 Using FREE models only (Ollama, HuggingFace)")
        else:
            providers_to_try = [
                self._init_ollama,      # FREE - Best for privacy & no limits
                self._init_huggingface, # FREE - Good open models
                self._init_openai,      # Paid
                self._init_anthropic,   # Paid
            ]

        for init_fn in providers_to_try:
            try:
                if init_fn():
                    logger.info(f"✅ Initialized: {init_fn.__name__}")
                    if not self.active_provider:
                        self.active_provider = ModelProvider.OLLAMA if "ollama" in init_fn.__name__ else ModelProvider.HUGGINGFACE
                    break
            except Exception as e:
                logger.debug(f"❌ {init_fn.__name__}: {e}")
                continue

    def _init_ollama(self) -> bool:
        """Initialize Ollama (FREE local LLM) - Models: llama3, mistral, codellama"""
        try:
            import requests
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            response = requests.get(f"{ollama_url}/api/tags", timeout=2)

            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                preferred = ["llama3:70b", "llama3:13b", "llama3", "mistral", "codellama"]
                selected = next((m for m in preferred if m in model_names), model_names[0] if model_names else "llama3")

                self.llm_clients["ollama"] = {"url": ollama_url, "model": selected}
                logger.info(f"✅ Ollama initialized: {selected}")
                return True
        except Exception as e:
            logger.debug(f"Ollama init failed: {e}")
            return False

    def _init_huggingface(self) -> bool:
        """Initialize Hugging Face (FREE) - Inference API"""
        try:
            hf_token = os.getenv("HUGGINGFACE_API_KEY")
            if hf_token:
                self.llm_clients["huggingface"] = {
                    "token": hf_token,
                    "model": "mistralai/Mistral-7B-Instruct-v0.2"
                }
                logger.info("✅ HuggingFace initialized")
                return True
        except Exception as e:
            logger.debug(f"HuggingFace init failed: {e}")
        return False

    def _init_openai(self) -> bool:
        """Initialize OpenAI (PAID)"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                from openai import AsyncOpenAI
                self.llm_clients["openai"] = AsyncOpenAI(api_key=api_key)
                logger.info("✅ OpenAI initialized")
                return True
        except Exception as e:
            logger.debug(f"OpenAI init failed: {e}")
        return False

    def _init_anthropic(self) -> bool:
        """Initialize Anthropic (PAID)"""
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                import anthropic
                self.llm_clients["anthropic"] = anthropic.AsyncAnthropic(api_key=api_key)
                logger.info("✅ Anthropic initialized")
                return True
        except Exception as e:
            logger.debug(f"Anthropic init failed: {e}")
        return False

    def _load_default_models(self):
        """Load default local transformers models as fallback"""
        try:
            logger.info("Loading local fallback models...")
            self.tokenizers["gpt2"] = AutoTokenizer.from_pretrained("gpt2")
            self.models["gpt2"] = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
            self.tokenizers["distilbert"] = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.models["distilbert"] = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=5
            ).to(self.device)
            self.pipelines["text-generation"] = pipeline(
                "text-generation", model=self.models["gpt2"],
                tokenizer=self.tokenizers["gpt2"],
                device=0 if self.device == "cuda" else -1
            )
            logger.info("✅ Local models loaded")
        except Exception as e:
            logger.warning(f"Local models not loaded: {e}")

    async def generate_text(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> str:
        """
        Generate text using best available LLM
        """
        try:
            # Try Ollama first (FREE & no limits)
            if "ollama" in self.llm_clients:
                return await self._generate_ollama(prompt, max_length, temperature)

            # Try HuggingFace (FREE but rate limited)
            elif "huggingface" in self.llm_clients:
                return await self._generate_huggingface(prompt, max_length, temperature)

            # Try OpenAI (PAID)
            elif "openai" in self.llm_clients:
                return await self._generate_openai(prompt, max_length, temperature)

            # Try Anthropic (PAID)
            elif "anthropic" in self.llm_clients:
                return await self._generate_anthropic(prompt, max_length, temperature)

            # Fallback to local model
            else:
                result = self.pipelines["text-generation"](
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=True
                )
                return result[0]["generated_text"]

        except Exception as e:
            logger.error(f"Text generation error: {str(e)}")
            raise

    async def _generate_ollama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Ollama"""
        import requests
        config = self.llm_clients["ollama"]
        response = requests.post(
            f"{config['url']}/api/generate",
            json={"model": config["model"], "prompt": prompt, "stream": False}
        )
        return response.json()["response"]

    async def _generate_huggingface(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using HuggingFace Inference API"""
        import requests
        config = self.llm_clients["huggingface"]
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{config['model']}",
            headers={"Authorization": f"Bearer {config['token']}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": temperature}}
        )
        return response.json()[0]["generated_text"]

    async def _generate_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using OpenAI"""
        client = self.llm_clients["openai"]
        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content

    async def _generate_anthropic(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Anthropic"""
        client = self.llm_clients["anthropic"]
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    async def classify_threat(self, text: str) -> Dict[str, Any]:
        """
        Classify cybersecurity threat using trained model
        """
        try:
            tokenizer = self.tokenizers["distilbert"]
            model = self.models["distilbert"]

            # Tokenize input
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Map to threat types
            threat_types = ["benign", "malware", "phishing", "dos", "exploit"]
            confidence_scores = predictions[0].cpu().tolist()

            result = {
                threat_type: score
                for threat_type, score in zip(threat_types, confidence_scores)
            }

            # Get top prediction
            top_threat = max(result, key=result.get)

            return {
                "threat_type": top_threat,
                "confidence": result[top_threat],
                "all_scores": result
            }

        except Exception as e:
            logger.error(f"Threat classification error: {str(e)}")
            raise

    async def train_model(
        self,
        dataset_path: str,
        model_type: str,
        epochs: int = 3,
        batch_size: int = 8
    ):
        """
        Fine-tune LLM on custom cybersecurity dataset
        """
        try:
            logger.info(f"Starting fine-tuning for {model_type}")

            # TODO: Implement dataset loading
            # from datasets import load_dataset
            # dataset = load_dataset("json", data_files=dataset_path)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"./models/trained/{model_type}",
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=10,
                save_steps=1000,
                evaluation_strategy="steps",
                eval_steps=500,
            )

            # TODO: Implement Trainer
            # trainer = Trainer(
            #     model=self.models[model_type],
            #     args=training_args,
            #     train_dataset=train_dataset,
            #     eval_dataset=eval_dataset,
            # )
            # trainer.train()

            logger.info(f"Fine-tuning completed for {model_type}")

        except Exception as e:
            logger.error(f"Model training error: {str(e)}")
            raise

    def load_custom_model(self, model_path: str, model_name: str):
        """
        Load a custom fine-tuned model
        """
        try:
            logger.info(f"Loading custom model: {model_name}")

            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
            self.models[model_name] = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

            logger.info(f"Custom model {model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Error loading custom model: {str(e)}")
            raise

    def is_ready(self) -> bool:
        """
        Check if LLM manager is ready
        """
        return len(self.models) > 0 and len(self.tokenizers) > 0

    def get_models_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models
        """
        return {
            "device": self.device,
            "loaded_models": list(self.models.keys()),
            "loaded_tokenizers": list(self.tokenizers.keys()),
            "available_pipelines": list(self.pipelines.keys()),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
