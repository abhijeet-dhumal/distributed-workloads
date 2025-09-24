#!/usr/bin/env python3
"""
Ray Data Synthetic Data Generation Job
Advanced distributed SDG using Ray Data pipelines for scalable, fault-tolerant processing
"""

import os
import json
import ray
import ray.data
import torch
import warnings
import time
import argparse
import numpy as np
import signal
import threading
from typing import List, Dict, Optional, Any, Iterator
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
import re

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

# Model configurations
PRIMARY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL = "microsoft/DialoGPT-medium"
BACKUP_MODEL = "microsoft/DialoGPT-small"

MODEL_CONFIGS = {
    "Qwen/Qwen2.5-1.5B-Instruct": {"size": "1.5B", "memory_mb": 6000},
    "microsoft/DialoGPT-medium": {"size": "355M", "memory_mb": 2000},
    "microsoft/DialoGPT-small": {"size": "117M", "memory_mb": 1000}
}

class CheckpointManager:
    """Manages checkpointing and resume functionality"""
    
    def __init__(self, output_dir: str, save_every: int = 5):
        self.output_dir = output_dir
        self.save_every = save_every
        self.checkpoint_file = os.path.join(output_dir, "checkpoint.json")
        self.dataset_file = os.path.join(output_dir, "synthetic_dataset.json")
        self.metadata_file = os.path.join(output_dir, "dataset_metadata.json")
        self.processed_seeds = set()
        self.current_data = []
        self.checkpoint_count = 0
        self.lock = threading.Lock()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def load_checkpoint(self) -> Dict:
        """Load existing checkpoint if available"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                
                # Load processed seeds
                self.processed_seeds = set(checkpoint.get('processed_seeds', []))
                self.checkpoint_count = checkpoint.get('checkpoint_count', 0)
                
                # Load existing data
                if os.path.exists(self.dataset_file):
                    with open(self.dataset_file, 'r') as f:
                        self.current_data = json.load(f)
                
                print(f"Loaded checkpoint: {len(self.processed_seeds)} seeds processed, {len(self.current_data)} samples saved")
                return checkpoint
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")
        
        return {}
    
    def save_checkpoint(self, processed_seeds: set, total_expected: int, force: bool = False):
        """Save current progress to checkpoint"""
        with self.lock:
            self.processed_seeds.update(processed_seeds)
            
            if force or len(self.processed_seeds) % self.save_every == 0:
                checkpoint = {
                    'processed_seeds': list(self.processed_seeds),
                    'checkpoint_count': self.checkpoint_count + 1,
                    'total_expected': total_expected,
                    'timestamp': time.time(),
                    'progress_percentage': (len(self.processed_seeds) / total_expected) * 100 if total_expected > 0 else 0
                }
                
                try:
                    with open(self.checkpoint_file, 'w') as f:
                        json.dump(checkpoint, f, indent=2)
                    
                    self.checkpoint_count += 1
                    print(f"Checkpoint saved: {len(self.processed_seeds)}/{total_expected} seeds processed ({checkpoint['progress_percentage']:.1f}%)")
                except Exception as e:
                    print(f"Warning: Failed to save checkpoint: {e}")
    
    def save_batch_data(self, batch_data: List[Dict]):
        """Save batch data incrementally"""
        with self.lock:
            self.current_data.extend(batch_data)
            
            try:
                with open(self.dataset_file, 'w') as f:
                    json.dump(self.current_data, f, indent=2, cls=NumpyEncoder)
                
                # Update metadata
                metadata = {
                    'total_samples': len(self.current_data),
                    'high_quality_count': sum(1 for item in self.current_data if item.get('overall_quality', 0) >= 0.3),
                    'last_update': time.time(),
                    'checkpoint_count': self.checkpoint_count
                }
                
                if metadata['total_samples'] > 0:
                    metadata['quality_pass_rate'] = (metadata['high_quality_count'] / metadata['total_samples']) * 100
                    metadata['avg_quality_score'] = np.mean([item.get('overall_quality', 0) for item in self.current_data])
                
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, cls=NumpyEncoder)
                
                print(f"Saved batch: {len(batch_data)} new samples (Total: {len(self.current_data)})")
                
            except Exception as e:
                print(f"Warning: Failed to save batch data: {e}")
    
    def get_remaining_seeds(self, all_seed_ids: List[int]) -> List[int]:
        """Get list of seeds that haven't been processed yet"""
        return [seed_id for seed_id in all_seed_ids if seed_id not in self.processed_seeds]
    
    def is_seed_processed(self, seed_id: int) -> bool:
        """Check if a seed has already been processed"""
        return seed_id in self.processed_seeds


class ModelInferenceCallable:
    """Ray Data callable class for distributed model inference with checkpointing"""
    
    def __init__(self, model_name: str = PRIMARY_MODEL, variations_per_seed: int = 1, output_dir: str = "/tmp/synthetic_data", processed_seeds: set = None):
        self.model_name = model_name
        self.variations_per_seed = variations_per_seed
        self.output_dir = output_dir
        self.processed_seeds = processed_seeds or set()
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def __call__(self, batch: Dict[str, List]) -> Dict[str, List]:
        """Process a batch of seed samples to generate synthetic data"""
        if self.model is None:
            self._initialize_model()
        
        print(f"[Batch] Processing batch with {len(batch.get('seed_samples', []))} seeds")
        
        results = {
            "questions": [],
            "answers": [],
            "sources": [],
            "difficulties": [],
            "concepts": [],
            "quality_scores": [],
            "model_confidences": [],
            "seed_ids": [],
            "variation_ids": []
        }
        
        processed_seed_ids = set()
        batch_data = []
        
        # Process each seed in the batch
        for i, seed_sample in enumerate(batch["seed_samples"]):
            seed_id = batch.get("seed_ids", [i])[i]
            
            # Skip if already processed (for resume functionality)
            if seed_id in self.processed_seeds:
                print(f"[Batch] Skipping already processed seed {seed_id}")
                continue
            
            print(f"[Batch] Processing seed {seed_id}: {seed_sample.get('question', 'No question')[:50]}...")
            
            # Generate variations for this seed
            for var_id in range(self.variations_per_seed):
                try:
                    generated = self._generate_variation(seed_sample, var_id)
                    if generated:
                        print(f"[Batch] Generated variation {var_id} for seed {seed_id}")
                        if self._quick_quality_check(generated):
                            print(f"[Batch] Variation {var_id} passed quality check")
                        else:
                            print(f"[Batch] Variation {var_id} failed quality check")
                    else:
                        print(f"[Batch] Failed to generate variation {var_id} for seed {seed_id}")
                    
                    if generated and self._quick_quality_check(generated):
                        quality_scores = self._assess_quality(generated)
                        
                        # Add to results for Ray Data pipeline
                        results["questions"].append(str(generated["question"]))
                        results["answers"].append(str(generated["answer"]))
                        results["sources"].append("ray_data_sdg_qwen")
                        results["difficulties"].append(str(generated.get("difficulty", "medium")))
                        results["concepts"].append("arithmetic,word_problems")
                        results["quality_scores"].append(float(quality_scores["overall_quality"]))
                        results["model_confidences"].append(float(generated.get("confidence", 0.5)))
                        results["seed_ids"].append(int(seed_id))
                        results["variation_ids"].append(int(var_id))
                        
                        # Also add to batch data for immediate saving
                        batch_item = {
                            "question": str(generated["question"]),
                            "answer": str(generated["answer"]),
                            "source": "ray_data_sdg_qwen",
                            "difficulty": str(generated.get("difficulty", "medium")),
                            "concepts": ["arithmetic", "word_problems"],
                            "overall_quality": float(quality_scores["overall_quality"]),
                            "model_confidence": float(generated.get("confidence", 0.5)),
                            "seed_id": int(seed_id),
                            "variation_id": int(var_id)
                        }
                        batch_data.append(batch_item)
                        
                except Exception as e:
                    print(f"[Batch] Error processing seed {seed_id}, variation {var_id}: {e}")
                    continue
            
            processed_seed_ids.add(seed_id)
        
        # Save batch data incrementally
        if batch_data:
            self._save_batch_data(batch_data)
        
        return results
    
    def _initialize_model(self):
        """Initialize model and tokenizer on worker"""
        print(f"[Worker] Loading model: {self.model_name}")
        
        cache_dir = self._get_cache_directory()
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                cache_dir=cache_dir
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model
            model_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                dtype=model_dtype,
                cache_dir=cache_dir,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print(f"[Worker] Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"[Worker] Error loading model {self.model_name}: {e}")
            raise
    
    def _get_cache_directory(self) -> str:
        """Get cache directory for model storage"""
        possible_paths = [
            "/shared/cache",
            os.path.expanduser("~/.cache"),
            "/tmp/.cache"
        ]
        
        for path in possible_paths:
            if os.path.exists(os.path.dirname(path)):
                os.makedirs(path, exist_ok=True)
                return path
        
        return "/tmp/.cache"
    
    def _generate_variation(self, seed_sample: Dict, var_id: int) -> Optional[Dict]:
        """Generate a single variation from a seed sample"""
        difficulties = ["easy", "medium", "hard"]
        difficulty = difficulties[var_id % len(difficulties)]
        
        prompt = self._create_variation_prompt(seed_sample, difficulty)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Parse response
            parsed = self._parse_response(response)
            if parsed:
                parsed["difficulty"] = difficulty
                return parsed
            
        except Exception as e:
            print(f"[Worker] Generation error: {e}")
        
        return None
    
    def _create_variation_prompt(self, seed_sample: Dict, difficulty: str = "medium") -> str:
        """Create a prompt for generating variations"""
        seed_question = seed_sample["question"]
        seed_answer = seed_sample["answer"]
        
        prompt = f"""<|im_start|>system
You are a math problem generator. Create a new {difficulty} math word problem inspired by the example below. Return ONLY a JSON object with "question" and "answer" fields.

Example:
Question: {seed_question}
Answer: {seed_answer}

Generate a similar but different problem with a COMPLETE step-by-step solution that shows all calculations and arrives at a final numerical answer.
<|im_end|>
<|im_start|>user
Create a new {difficulty} math problem:
<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def _parse_response(self, response: str) -> Optional[Dict]:
        """Parse model response into structured format"""
        try:
            response = response.strip()
            
            # Try JSON parsing first
            if response.startswith('{') and response.endswith('}'):
                import json
                return json.loads(response)
            
            # Handle Qwen format: starts with quoted question
            if response.startswith('"') and '"answer":' in response:
                if not response.startswith('{'):
                    response = "{" + response
                if not response.endswith('}'):
                    response = response + "}"
                
                import json
                return json.loads(response)
            
            # Try to find JSON-like content within the response
            import re
            json_match = re.search(r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    import json
                    return json.loads(json_match.group(0))
                except:
                    pass
            
            # Fallback regex parsing
            return self._fallback_parse(response)
            
        except Exception as e:
            return self._fallback_parse(response)
    
    def _fallback_parse(self, response: str) -> Optional[Dict]:
        """Fallback parsing for non-JSON responses"""
        import re
        
        # Try to extract question and answer with improved regex that handles multi-line content
        question_match = re.search(r'"question":\s*"([^"]*(?:\\.[^"]*)*?)"', response, re.DOTALL)
        answer_match = re.search(r'"answer":\s*"([^"]*(?:\\.[^"]*)*?)"', response, re.DOTALL)
        
        if question_match and answer_match:
            # Clean up the extracted text by handling escaped quotes
            question = question_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            answer = answer_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            
            return {
                "question": question,
                "answer": answer,
                "confidence": 0.7
            }
        
        # Try alternative patterns for question and answer
        alt_question_match = re.search(r'Question:\s*(.+?)(?=Answer:|$)', response, re.DOTALL | re.IGNORECASE)
        alt_answer_match = re.search(r'Answer:\s*(.+?)(?=Question:|$)', response, re.DOTALL | re.IGNORECASE)
        
        if alt_question_match and alt_answer_match:
            return {
                "question": alt_question_match.group(1).strip(),
                "answer": alt_answer_match.group(1).strip(),
                "confidence": 0.6
            }
        
        # If response looks like a math problem, treat it as question
        if any(keyword in response.lower() for keyword in ["how many", "what is", "calculate", "find"]):
            return {
                "question": response.strip(),
                "answer": "This is a mathematical word problem that requires calculation.",
                "confidence": 0.5
            }
        
        return None
    
    def _quick_quality_check(self, generated: Dict) -> bool:
        """Quick quality check for generated content"""
        if not generated or not generated.get("question") or not generated.get("answer"):
            return False
        
        # Ensure question and answer are strings before calling .lower()
        question = str(generated["question"]).lower()
        answer = str(generated["answer"]).lower()
        
        # Check for mathematical content
        math_indicators = ["calculate", "solve", "find", "how many", "total", "cost", "price"]
        has_math = any(indicator in question for indicator in math_indicators)
        
        # Check minimum length and completeness
        min_length = len(question) >= 20 and len(answer) >= 30  # Increased minimum answer length
        
        # Check if answer seems complete (contains numbers or calculation words)
        has_calculation = any(word in answer for word in ["=", "total", "result", "answer is", "solution"])
        has_numbers = any(char.isdigit() for char in answer)
        
        # Avoid incomplete answers
        incomplete_indicators = ["let's", "step by step:", "to determine", "we need to", "first,"]
        is_incomplete = any(indicator in answer.lower() and len(answer) < 100 for indicator in incomplete_indicators)
        
        return has_math and min_length and (has_calculation or has_numbers) and not is_incomplete
    
    def _assess_quality(self, generated: Dict) -> Dict[str, float]:
        """Assess quality of generated content"""
        question = str(generated.get("question", ""))
        answer = str(generated.get("answer", ""))
        
        # Mathematical content score
        math_keywords = ["calculate", "solve", "find", "total", "sum", "difference", "product"]
        math_score = min(sum(1 for kw in math_keywords if kw in question.lower()) * 0.2, 1.0)
        
        # Answer quality score
        answer_indicators = ["step", "first", "then", "therefore", "solution"]
        answer_score = min(sum(1 for ind in answer_indicators if ind in answer.lower()) * 0.2, 1.0)
        
        # Length and structure score
        structure_score = 0.5
        if len(question.split()) >= 10:
            structure_score += 0.2
        if len(answer.split()) >= 5:
            structure_score += 0.2
        if any(char.isdigit() for char in answer):
            structure_score += 0.1
        
        overall_quality = (math_score + answer_score + structure_score) / 3
        
        return {
            "mathematical_content": math_score,
            "answer_quality": answer_score,
            "structure_quality": structure_score,
            "overall_quality": overall_quality
        }
    
    def _save_batch_data(self, batch_data: List[Dict]):
        """Save batch data incrementally"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Load existing data
            dataset_path = os.path.join(self.output_dir, "synthetic_dataset.json")
            existing_data = []
            if os.path.exists(dataset_path):
                try:
                    with open(dataset_path, "r") as f:
                        existing_data = json.load(f)
                except:
                    pass
            
            # Append new results
            existing_data.extend(batch_data)
            
            # Save updated data
            with open(dataset_path, "w") as f:
                json.dump(existing_data, f, indent=2, cls=NumpyEncoder)
            
            # Update metadata
            metadata = {
                "total_samples": len(existing_data),
                "high_quality_count": sum(1 for item in existing_data if item.get("overall_quality", 0) >= 0.3),
                "last_update": time.time(),
                "batch_count": len(batch_data)
            }
            
            if metadata["total_samples"] > 0:
                metadata["quality_pass_rate"] = (metadata["high_quality_count"] / metadata["total_samples"]) * 100
                metadata["avg_quality_score"] = np.mean([item.get("overall_quality", 0) for item in existing_data])
            
            metadata_path = os.path.join(self.output_dir, "dataset_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, cls=NumpyEncoder)
            
            print(f"Saved batch: {len(batch_data)} new samples (Total: {len(existing_data)})")
            
        except Exception as e:
            print(f"Warning: Failed to save batch data: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Ray Data Synthetic Data Generation for Mathematical Word Problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode - generate few samples quickly
  python ray_data_sdg_job.py --test-mode
  
  # Production mode - generate full dataset
  python ray_data_sdg_job.py
  
  # Custom configuration
  python ray_data_sdg_job.py --seeds 100 --variations 2 --batch-size 16
        """
    )
    
    parser.add_argument(
        "--test-mode", 
        action="store_true",
        help="Enable test mode with minimal samples for quick testing"
    )
    
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Number of seed samples to use (overrides test/production defaults)"
    )
    
    parser.add_argument(
        "--variations",
        type=int,
        default=None,
        help="Number of variations per seed (overrides test/production defaults)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for Ray Data processing"
    )
    
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=None,
        help="Quality threshold for filtering (0.0-1.0)"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default="/tmp/synthetic_data",
        help="Output path for generated dataset"
    )
    
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs to use for Ray Data processing (default: auto-detect based on mode)"
    )
    
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N processed seeds (default: 5)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint if available"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for checkpoint files (default: same as output-path)"
    )
    
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU-only execution even if GPUs are available"
    )
    
    return parser.parse_args()


def create_seed_dataset(num_seeds: int, cache_dir: str) -> ray.data.Dataset:
    """Create Ray Dataset from GSM8K seed samples"""
    print("Loading GSM8K dataset...")
    gsm8k_dataset = load_dataset("gsm8k", "main", cache_dir=f"{cache_dir}/datasets")
    print(f"Dataset loaded: {len(gsm8k_dataset['train'])} train samples")
    
    # Prepare seed samples
    train_data = gsm8k_dataset["train"]
    seed_samples = []
    
    for i in range(min(num_seeds, len(train_data))):
        sample = train_data[i]
        seed_samples.append({
            "seed_samples": {
                "question": sample["question"],
                "answer": sample["answer"]
            },
            "seed_ids": i
        })
    
    print(f"Created {len(seed_samples)} seed samples")
    
    # Create Ray Dataset
    return ray.data.from_items(seed_samples)


def create_seed_dataset_filtered(seed_ids: List[int], cache_dir: str) -> ray.data.Dataset:
    """Create Ray Dataset from GSM8K seed samples for specific seed IDs"""
    print(f"Loading GSM8K dataset for {len(seed_ids)} specific seeds...")
    gsm8k_dataset = load_dataset("gsm8k", "main", cache_dir=f"{cache_dir}/datasets")
    print(f"Dataset loaded: {len(gsm8k_dataset['train'])} train samples")
    
    # Prepare seed samples for specific IDs
    train_data = gsm8k_dataset["train"]
    seed_samples = []
    
    for seed_id in seed_ids:
        if seed_id < len(train_data):
            sample = train_data[seed_id]
            seed_samples.append({
                "seed_samples": {
                    "question": sample["question"],
                    "answer": sample["answer"]
                },
                "seed_ids": seed_id
            })
    
    print(f"Created {len(seed_samples)} filtered seed samples")
    
    # Create Ray Dataset
    return ray.data.from_items(seed_samples)


def quality_filter(batch: Dict[str, List]) -> Dict[str, List]:
    """Filter batch based on quality scores"""
    print(f"[Filter] Input batch has {len(batch.get('quality_scores', []))} items")
    
    filtered_indices = []
    
    for i, quality_score in enumerate(batch["quality_scores"]):
        print(f"[Filter] Item {i}: quality_score = {quality_score}")
        if quality_score >= 0.3:  # Lowered threshold for test mode
            filtered_indices.append(i)
    
    print(f"[Filter] {len(filtered_indices)} items passed quality filter")
    
    # Filter all fields based on quality
    filtered_batch = {}
    for key, values in batch.items():
        filtered_batch[key] = [values[i] for i in filtered_indices]
    
    return filtered_batch


def format_for_output(batch: Dict[str, List]) -> Dict[str, List]:
    """Format batch for final output"""
    formatted_items = []
    
    for i in range(len(batch["questions"])):
        item = {
            "question": str(batch["questions"][i]),
            "answer": str(batch["answers"][i]),
            "source": str(batch["sources"][i]),
            "difficulty": str(batch["difficulties"][i]),
            "concepts": batch["concepts"][i].split(",") if isinstance(batch["concepts"][i], str) else ["arithmetic", "word_problems"],
            "overall_quality": float(batch["quality_scores"][i]),
            "model_confidence": float(batch["model_confidences"][i]),
            "seed_id": int(batch["seed_ids"][i]),
            "variation_id": int(batch["variation_ids"][i])
        }
        formatted_items.append(item)
    
    return {"items": formatted_items}


def save_dataset(dataset: ray.data.Dataset, output_path: str, metadata: Dict):
    """Save dataset and metadata"""
    os.makedirs(output_path, exist_ok=True)
    
    # Collect all items
    all_items = []
    for batch in dataset.iter_batches(batch_size=None):
        all_items.extend(batch["items"])
    
    # Split into train/test (80/20)
    random.shuffle(all_items)
    split_idx = int(len(all_items) * 0.8)
    
    final_dataset = {
        "train": all_items[:split_idx],
        "test": all_items[split_idx:],
        "metadata": metadata
    }
    
    # Save dataset
    dataset_path = os.path.join(output_path, "synthetic_dataset.json")
    with open(dataset_path, "w") as f:
        json.dump(final_dataset, f, indent=2, cls=NumpyEncoder)
    
    # Save metadata
    metadata_path = os.path.join(output_path, "dataset_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)
    
    print(f"Dataset saved: {len(final_dataset['train'])} train / {len(final_dataset['test'])} test")
    print(f"Saved to: {dataset_path}")
    print(f"Metadata saved to: {metadata_path}")


def setup_signal_handlers(checkpoint_manager: CheckpointManager, total_expected: int):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}. Saving checkpoint and shutting down gracefully...")
        checkpoint_manager.save_checkpoint(checkpoint_manager.processed_seeds, total_expected, force=True)
        print("Checkpoint saved. Exiting...")
        if ray.is_initialized():
            ray.shutdown()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main function using Ray Data pipeline with checkpointing"""
    args = parse_args()
    
    print("Starting Ray Data distributed synthetic data generation with checkpointing...")
    print(f"Mode: {'TEST' if args.test_mode else 'PRODUCTION'}")
    
    # Setup checkpoint directory
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else args.output_path
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir, args.save_every)
    
    # Load existing checkpoint if resuming
    checkpoint_data = {}
    if args.resume:
        print("Attempting to resume from checkpoint...")
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            print(f"Resuming from checkpoint with {len(checkpoint_manager.processed_seeds)} seeds already processed")
        else:
            print("No checkpoint found, starting fresh")
    else:
        print("Starting fresh (not resuming from checkpoint)")
    
    # Initialize Ray (handle both standalone and Ray job execution)
    if not ray.is_initialized():
        print("Initializing Ray...")
        
        # Check if we're running inside a Ray job by looking for Ray runtime environment
        import os
        is_ray_job = os.environ.get('RAY_JOB_ID') is not None or os.environ.get('RAY_ADDRESS') is not None
        
        if is_ray_job:
            # Running as part of a Ray job - connect without resource specifications
            ray.init()
            print("Ray initialized in cluster mode (running as Ray job)")
        else:
            # Running standalone - initialize with resource limits
            ray.init(num_cpus=min(8, os.cpu_count()))
            print("Ray initialized in standalone mode")
    else:
        print("Ray already initialized")
    
    print(f"Connected to Ray cluster: {ray.cluster_resources()}")
    
    # Configure based on mode
    if args.test_mode:
        default_seeds = 2
        default_variations = 1
        default_batch_size = 1
        default_quality_threshold = 0.3
        default_num_cpus = 2  # Conservative for testing
        print("TEST MODE: Using minimal samples for quick testing")
    else:
        default_seeds = 50
        default_variations = 3
        default_batch_size = 8
        default_quality_threshold = 0.6
        default_num_cpus = 4  # Reasonable for production
        print("PRODUCTION MODE: Generating full-scale dataset")
    
    # Apply user overrides
    num_seeds = args.seeds if args.seeds is not None else default_seeds
    variations_per_seed = args.variations if args.variations is not None else default_variations
    batch_size = args.batch_size if args.batch_size is not None else default_batch_size
    quality_threshold = args.quality_threshold if args.quality_threshold is not None else default_quality_threshold
    num_cpus = args.num_cpus if args.num_cpus is not None else default_num_cpus
    
    total_expected_seeds = num_seeds
    
    print(f"Configuration:")
    print(f"  - Seeds: {num_seeds}")
    print(f"  - Variations per seed: {variations_per_seed}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Quality threshold: {quality_threshold}")
    print(f"  - CPUs to use: {num_cpus}")
    print(f"  - Save every: {args.save_every} seeds")
    print(f"  - Checkpoint dir: {checkpoint_dir}")
    print(f"  - Expected total: {num_seeds} x {variations_per_seed} = {num_seeds * variations_per_seed} problems")
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers(checkpoint_manager, total_expected_seeds)
    
    # Get cache directory
    cache_dir = os.path.expanduser("~/.cache")
    if os.path.exists("/shared/cache"):
        cache_dir = "/shared/cache"
    
    try:
        # Create seed dataset
        all_seed_ids = list(range(num_seeds))
        
        # Filter out already processed seeds if resuming
        if args.resume and checkpoint_manager.processed_seeds:
            remaining_seed_ids = checkpoint_manager.get_remaining_seeds(all_seed_ids)
            print(f"Resuming: {len(remaining_seed_ids)} seeds remaining out of {num_seeds} total")
            if not remaining_seed_ids:
                print("All seeds already processed! Nothing to do.")
                return
        else:
            remaining_seed_ids = all_seed_ids
        
        # Create seed dataset with remaining seeds
        seed_ds = create_seed_dataset_filtered(remaining_seed_ids, cache_dir)
        
        print(f"\nStarting Ray Data pipeline with checkpointing...")
        start_time = time.time()
        
        # Ray Data Pipeline with checkpoint manager
        # Determine resource allocation based on cluster resources (not just torch.cuda.is_available)
        cluster_resources = ray.cluster_resources()
        available_gpus = cluster_resources.get("GPU", 0)
        available_cpus = cluster_resources.get("CPU", 0)
        
        print(f"Cluster resources: CPU={available_cpus}, GPU={available_gpus}")
        
        if args.cpu_only:
            # Force CPU-only mode
            compute_resources = {"num_cpus": min(num_cpus, int(available_cpus))}
            print(f"Forced CPU-only mode: {compute_resources}")
        elif available_gpus > 0 and torch.cuda.is_available():
            # Use GPU only if cluster actually has GPUs available
            compute_resources = {"num_gpus": 1}
            print("Using GPU resources for model inference")
        else:
            # Use CPUs - safer fallback
            compute_resources = {"num_cpus": min(num_cpus, int(available_cpus))}
            print(f"Using CPU resources for model inference: {compute_resources}")
        
        results_ds = (seed_ds
            .map_batches(
                ModelInferenceCallable(PRIMARY_MODEL, variations_per_seed, checkpoint_dir, checkpoint_manager.processed_seeds),
                batch_size=batch_size,
                concurrency=1,  # Process one batch at a time to control resource usage
                **compute_resources
            )
            .filter(lambda batch: len(batch["questions"]) > 0)  # Remove empty batches
            .map_batches(quality_filter)
            .map_batches(format_for_output)
        )
        
        # Execute pipeline and collect results with checkpointing
        print("Executing Ray Data pipeline with incremental saving...")
        
        # Count total results
        total_generated = 0
        high_quality_count = 0
        quality_scores = []
        processed_batches = 0
        
        for batch in results_ds.iter_batches(batch_size=None):
            items = batch["items"]
            total_generated += len(items)
            processed_batches += 1
            
            # Extract seed IDs from this batch for checkpoint tracking
            batch_seed_ids = set()
            for item in items:
                if item["overall_quality"] >= quality_threshold:
                    high_quality_count += 1
                quality_scores.append(item["overall_quality"])
                batch_seed_ids.add(item["seed_id"])
            
            # Update checkpoint with processed seeds
            if batch_seed_ids:
                checkpoint_manager.save_checkpoint(batch_seed_ids, total_expected_seeds)
            
            print(f"Processed batch {processed_batches}: {len(items)} items, {len(batch_seed_ids)} seeds")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Create metadata
        metadata = {
            "total_generated": total_generated,
            "high_quality_count": high_quality_count,
            "quality_pass_rate": (high_quality_count / total_generated * 100) if total_generated > 0 else 0,
            "quality_threshold": quality_threshold,
            "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "processing_time_seconds": processing_time,
            "model_used": PRIMARY_MODEL,
            "generation_method": "ray_data_distributed",
            "ray_data_features": [
                "map_batches_inference",
                "automatic_scaling",
                "fault_tolerance",
                "streaming_processing",
                "quality_filtering"
            ]
        }
        
        print("\n" + "="*60)
        print("RAY DATA SDG PIPELINE SUMMARY")
        print("="*60)
        print(f"Total problems generated: {total_generated}")
        print(f"High quality problems: {high_quality_count}")
        print(f"Quality pass rate: {metadata['quality_pass_rate']:.1f}%")
        print(f"Average quality score: {metadata['avg_quality_score']:.3f}")
        print(f"Processing time: {processing_time:.1f} seconds")
        print(f"Throughput: {total_generated/processing_time:.2f} problems/second")
        print("="*60)
        
        # Final checkpoint save
        checkpoint_manager.save_checkpoint(checkpoint_manager.processed_seeds, total_expected_seeds, force=True)
        
        # Save final results (already saved incrementally, but create final formatted version)
        if checkpoint_manager.current_data:
            save_final_dataset(checkpoint_manager.current_data, args.output_path, metadata)
        
        print(f"\nPipeline completed successfully!")
        print(f"Results saved to: {args.output_path}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
        
    except KeyboardInterrupt:
        print(f"\nPipeline interrupted by user")
        checkpoint_manager.save_checkpoint(checkpoint_manager.processed_seeds, total_expected_seeds, force=True)
        print("Progress saved to checkpoint")
    except Exception as e:
        print(f"Error in Ray Data pipeline: {e}")
        checkpoint_manager.save_checkpoint(checkpoint_manager.processed_seeds, total_expected_seeds, force=True)
        print("Progress saved to checkpoint before exit")
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()


def save_final_dataset(all_items: List[Dict], output_path: str, metadata: Dict):
    """Save final formatted dataset"""
    os.makedirs(output_path, exist_ok=True)
    
    # Split into train/test (80/20)
    random.shuffle(all_items)
    split_idx = int(len(all_items) * 0.8)
    
    final_dataset = {
        "train": all_items[:split_idx],
        "test": all_items[split_idx:],
        "metadata": metadata
    }
    
    # Save final dataset
    final_dataset_path = os.path.join(output_path, "final_synthetic_dataset.json")
    with open(final_dataset_path, "w") as f:
        json.dump(final_dataset, f, indent=2, cls=NumpyEncoder)
    
    print(f"Final dataset saved: {len(final_dataset['train'])} train / {len(final_dataset['test'])} test")
    print(f"Saved to: {final_dataset_path}")


if __name__ == "__main__":
    main()
