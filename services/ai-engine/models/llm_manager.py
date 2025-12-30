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

logger = logging.getLogger(__name__)

class LLMManager:
    """
    Manages multiple LLM models for different cybersecurity tasks
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.pipelines: Dict[str, Any] = {}

        logger.info(f"LLM Manager initialized on device: {self.device}")

        # Load default models
        self._load_default_models()

    def _load_default_models(self):
        """
        Load default pre-trained models for cybersecurity
        """
        try:
            # 1. GPT-2 based model for general text generation
            logger.info("Loading GPT-2 model for text generation...")
            self.tokenizers["gpt2"] = AutoTokenizer.from_pretrained("gpt2")
            self.models["gpt2"] = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)

            # 2. DistilBERT for classification (threat detection)
            logger.info("Loading DistilBERT for classification...")
            self.tokenizers["distilbert"] = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.models["distilbert"] = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=5  # Attack types: benign, malware, phishing, dos, exploit
            ).to(self.device)

            # 3. Create pipelines
            self.pipelines["text-generation"] = pipeline(
                "text-generation",
                model=self.models["gpt2"],
                tokenizer=self.tokenizers["gpt2"],
                device=0 if self.device == "cuda" else -1
            )

            logger.info("Default models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading default models: {str(e)}")

    async def generate_text(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> str:
        """
        Generate text using LLM
        """
        try:
            # Use the text generation pipeline
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
