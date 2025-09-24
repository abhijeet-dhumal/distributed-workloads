#!/usr/bin/env python3
"""
Synthetic Data Generation Job for Ray Cluster
"""

import os
import json
import ray
import torch
import warnings
import time
import signal
import argparse
from typing import List, Dict, Optional
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from threading import Timer
from tqdm import tqdm
import random

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


PRIMARY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL = "microsoft/DialoGPT-medium"
BACKUP_MODEL = "microsoft/DialoGPT-small"

MODEL_CONFIGS = {
    "Qwen/Qwen2.5-1.5B-Instruct": {"size": "1.5B", "memory_mb": 6000},
    "microsoft/DialoGPT-medium": {"size": "355M", "memory_mb": 2000},
    "microsoft/DialoGPT-small": {"size": "117M", "memory_mb": 1000}
}

def create_worker_class(use_gpu: bool = False):
    if use_gpu:
        @ray.remote(num_cpus=1, memory=8000, num_gpus=1)
        class GPUSDGWorker(SDGWorkerBase):
            pass
        return GPUSDGWorker
    else:
        @ray.remote(num_cpus=1, memory=6000, num_gpus=0)
        class CPUSDGWorker(SDGWorkerBase):
            pass
        return CPUSDGWorker

class SDGWorkerBase:
    def __init__(self, model_name: str = PRIMARY_MODEL, variations_per_seed: int = 1):
        self.model_name = model_name
        self.variations_per_seed = variations_per_seed
        config = MODEL_CONFIGS.get(model_name, {"size": "unknown", "memory_mb": 4000})
        
        print(f"[Worker] Loading model: {self.model_name} ({config['size']} params)")
        
        self.cache_dir = self._get_shared_cache_dir()
        print(f"[Worker] Using cache directory: {self.cache_dir}")
        
        try:
            self.tokenizer = self._load_with_timeout(
                lambda: AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, cache_dir=self.cache_dir),
                timeout_seconds=300, operation_name="tokenizer loading"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
            
            self.model = self._load_with_timeout(
                lambda: AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype=torch_dtype, device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True, cache_dir=self.cache_dir, low_cpu_mem_usage=True
                ),
                timeout_seconds=600, operation_name="model loading"
            )
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            self.device = device
            print(f"[Worker] Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"[Worker] Error loading model: {e}")
            raise
    
    def _get_shared_cache_dir(self) -> str:
        for cache_path in ["/shared/cache", "/tmp/.cache"]:
            try:
                os.makedirs(cache_path, exist_ok=True)
                test_file = os.path.join(cache_path, ".cache_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                return cache_path
            except (OSError, PermissionError):
                continue
        return "/tmp/.cache"
    
    def _load_with_timeout(self, load_func, timeout_seconds: int, operation_name: str):
        import threading
        import queue
        
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def load_worker():
            try:
                result_queue.put(load_func())
            except Exception as e:
                exception_queue.put(e)
        
        load_thread = threading.Thread(target=load_worker)
        load_thread.daemon = True
        load_thread.start()
        
        start_time = time.time()
        last_progress_time = start_time
        
        while load_thread.is_alive():
            elapsed = time.time() - start_time
            
            if time.time() - last_progress_time >= 30:
                print(f"[Worker] Still {operation_name}... ({elapsed:.0f}s elapsed)")
                last_progress_time = time.time()
            
            if elapsed > timeout_seconds:
                raise TimeoutError(f"Timeout during {operation_name}")
            
            time.sleep(1)
        
        if not exception_queue.empty():
            raise exception_queue.get()
        
        if not result_queue.empty():
            return result_queue.get()
        else:
            raise RuntimeError(f"No result from {operation_name}")
    
    def generate_math_problems(self, gsm8k_samples: List[Dict]) -> List[Dict]:
        all_results = []
        total_attempts = len(gsm8k_samples) * self.variations_per_seed
        
        pbar = tqdm(total=total_attempts, desc=f"[Worker] Generating problems", position=0, leave=False)
        
        for i, seed_sample in enumerate(gsm8k_samples):
            try:
                difficulties = ["easy", "medium", "hard"]
                for var_id in range(self.variations_per_seed):
                    difficulty = difficulties[var_id % len(difficulties)]
                    generated = self._generate_with_retry(seed_sample, difficulty, max_retries=3)
                    
                    if generated and self._quick_quality_check(generated):
                        quality_scores = self._assess_quality(generated)
                        
                        if quality_scores["overall_quality"] >= 0.4:
                            all_results.append({
                                "question": generated["question"],
                                "answer": generated["answer"],
                                "context": seed_sample["question"],
                                "source": "ray_sdg_qwen",
                                "seed_id": seed_sample.get("seed_id", i),
                                "variation_id": var_id,
                                "difficulty": generated.get("difficulty", "medium"),
                                "concepts": generated.get("concepts", []),
                                "model_confidence": generated.get("confidence", 0.5),
                                "quality_scores": quality_scores,
                                "overall_quality": quality_scores["overall_quality"]
                            })
                    
                    pbar.update(1)
                    
            except Exception as e:
                print(f"[Worker] Error processing seed {i}: {e}")
                pbar.update(self.variations_per_seed)
                continue
        
        pbar.close()
        return all_results
    
    def _create_variation_prompt(self, seed_sample: Dict, difficulty: str = "medium") -> str:
        return f"""<|im_start|>system
You are a math teacher. Create a new math problem similar to the example. Respond with valid JSON only.
<|im_end|>
<|im_start|>user
Example problem:
{seed_sample['question']}

Create a {difficulty} math problem with different numbers and context but similar concepts.

Format your response as valid JSON:
{{"question": "your new math problem", "answer": "step by step solution", "difficulty": "{difficulty}", "concepts": ["math", "topics"], "confidence": 0.9}}
<|im_end|>
<|im_start|>assistant
{{"question": """
    
    def _generate_with_retry(self, seed_sample: Dict, difficulty: str, max_retries: int = 3) -> Optional[Dict]:
        for attempt in range(max_retries):
            try:
                prompt = self._create_variation_prompt(seed_sample, difficulty)
                if attempt > 0:
                    prompt = self._add_prompt_variation(prompt, attempt)
                
                generated = self._generate_with_model(prompt)
                if generated and self._validate_math_problem(generated):
                    return generated
                
                if attempt < max_retries - 1:
                    time.sleep((2 ** attempt) * 0.1)
                    
            except Exception as e:
                print(f"[Worker] Generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                continue
        
        return None
    
    def _add_prompt_variation(self, prompt: str, attempt: int) -> str:
        """Add variation to prompt for retry attempts"""
        variations = [
            "\nIMPORTANT: Focus on creating a mathematically sound problem with clear numerical answer.",
            "\nREMINDER: Ensure the problem has specific numbers and a definitive solution.",
            "\nNOTE: Create a problem that requires calculation to solve, not just conceptual understanding."
        ]
        
        if attempt - 1 < len(variations):
            # Insert the variation before the <|im_end|> tag
            return prompt.replace("<|im_end|>", variations[attempt - 1] + "\n<|im_end|>")
        return prompt
    
    def _generate_with_model(self, prompt: str) -> Dict:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return self._parse_response(response)
        except Exception as e:
            print(f"[Worker] Generation error: {e}")
            return None
    
    def _parse_response(self, response: str) -> Dict:
        """Parse JSON response with robust error handling"""
        try:
            # Clean the response - remove any text before/after JSON
            response = response.strip()
            
            # Debug logging - show more of the raw response to debug parsing
            print(f"[Worker] Raw response: {response[:400]}{'...' if len(response) > 400 else ''}")
            
            # Remove common prefixes that models add (including Qwen-specific ones)
            prefixes_to_remove = ["```json", "```javascript", "```", "---", "json", "Here's the JSON:", "Here is the JSON:"]
            for prefix in prefixes_to_remove:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
            
            # Handle case where response starts with "question": instead of {
            if response.startswith('"question":'):
                response = "{" + response
                print(f"[Worker] Fixed missing opening brace: {response[:100]}...")
            
            # Also handle case where response has fields but no braces at all
            if ('"question":' in response and '"answer":' in response and 
                not response.strip().startswith('{') and not response.strip().endswith('}')):
                response = "{" + response.strip() + "}"
                print(f"[Worker] Added missing braces: {response[:100]}...")
            
            # Remove common suffixes
            suffixes_to_remove = ["```", "---", "<|im_end|>"]
            for suffix in suffixes_to_remove:
                if response.endswith(suffix):
                    response = response[:-len(suffix)].strip()
            
            # Find JSON boundaries - look for the first complete JSON object
            start_idx = response.find('{')
            if start_idx == -1:
                print(f"[Worker] No JSON found in response: {response[:100]}...")
                return self._fallback_parse(response)
            
            # Find the matching closing brace
            brace_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if brace_count != 0:
                print(f"[Worker] Unmatched braces in JSON response")
                return self._fallback_parse(response)
            
            json_str = response[start_idx:end_idx]
            
            # Parse JSON
            import json
            parsed = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["question", "answer"]
            if not all(field in parsed for field in required_fields):
                print(f"[Worker] Missing required fields in JSON: {list(parsed.keys())}")
                return self._fallback_parse(response)
            
            # Add default values for optional fields
            parsed.setdefault("difficulty", "medium")
            parsed.setdefault("concepts", [])
            parsed.setdefault("confidence", 0.5)
            
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"[Worker] JSON decode error: {e}")
            return self._fallback_parse(response)
        except Exception as e:
            print(f"[Worker] Parsing error: {e}")
            return None
    
    def _fallback_parse(self, response: str) -> Dict:
        """Fallback parsing for non-JSON responses, especially Qwen format"""
        try:
            # Handle Qwen's common format: "question": "...", "answer": "...", etc.
            import re
            
            # Try to extract using regex patterns (more flexible)
            # Handle case where response starts with just the question text in quotes
            question_match = re.search(r'"question":\s*"([^"]*(?:"[^"]*"[^"]*)*)"', response, re.DOTALL)
            if not question_match:
                # Try to match if response starts with quoted text before ", "answer":
                question_match = re.search(r'^"([^"]*(?:"[^"]*"[^"]*)*)",\s*"answer":', response, re.DOTALL)
            
            # Simpler answer matching - capture everything until the closing quote
            answer_match = re.search(r'"answer":\s*"([^"]*(?:\\.[^"]*)*)"', response, re.DOTALL)
            difficulty_match = re.search(r'"difficulty":\s*"([^"]+)"', response)
            concepts_match = re.search(r'"concepts":\s*\[([^\]]*)\]', response)
            confidence_match = re.search(r'"confidence":\s*([0-9.]+)', response)
            
            print(f"[Worker] Regex matches - Q: {bool(question_match)}, A: {bool(answer_match)}")
            
            if question_match and answer_match:
                # Extract concepts if found
                concepts = []
                if concepts_match:
                    concepts_str = concepts_match.group(1)
                    concepts = [c.strip().strip('"') for c in concepts_str.split(',') if c.strip()]
                
                answer_text = answer_match.group(1)
                print(f"[Worker] Extracted answer: {answer_text[:100]}{'...' if len(answer_text) > 100 else ''}")
                print(f"[Worker] Successfully parsed Q&A from regex")
                return {
                    "question": question_match.group(1),
                    "answer": answer_text,
                    "difficulty": difficulty_match.group(1) if difficulty_match else "medium",
                    "concepts": concepts,
                    "confidence": float(confidence_match.group(1)) if confidence_match else 0.5
                }
            
            # Fallback to line-by-line parsing
            lines = response.strip().split('\n')
            question = ""
            answer = ""
            
            for i, line in enumerate(lines):
                line = line.strip()
                if any(keyword in line.lower() for keyword in ["question:", "problem:", "q:"]):
                    question = line.split(':', 1)[-1].strip().strip('"')
                elif any(keyword in line.lower() for keyword in ["answer:", "solution:", "a:"]):
                    answer = '\n'.join(lines[i:]).split(':', 1)[-1].strip().strip('"')
                    break
            
            # If no structured parsing worked, treat the whole response as a question
            if not question and not answer:
                # Check if this looks like a math problem
                if any(keyword in response.lower() for keyword in ["how many", "what is", "calculate", "find", "total", "cost"]):
                    question = response.strip().strip('"').strip("'")
                    if question.endswith('",'):
                        question = question[:-2]
                    
                    # Generate a simple answer
                    import re
                    numbers = re.findall(r'\b\d+\b', question)
                    if len(numbers) >= 2:
                        if any(word in question.lower() for word in ["left", "remaining", "subtract"]):
                            nums = [int(n) for n in numbers]
                            result = nums[0] - sum(nums[1:])
                            answer = f"Starting with {nums[0]}, after subtracting {' and '.join(str(n) for n in nums[1:])}, we get {result}."
                        else:
                            result = sum(int(n) for n in numbers)
                            answer = f"Adding the numbers: {' + '.join(numbers)} = {result}"
                    else:
                        answer = "This is a mathematical word problem that requires calculation."
            
            if question and answer:
                return {
                    "question": question,
                    "answer": answer,
                    "difficulty": "medium",
                    "concepts": ["arithmetic", "word_problems"],
                    "confidence": 0.6
                }
        except Exception as e:
            print(f"[Worker] Fallback parsing error: {e}")
            return None
    
    def _validate_math_problem(self, generated: Dict) -> bool:
        if not generated or not generated.get("question") or not generated.get("answer"):
            return False
        
        question = generated["question"].lower()
        math_indicators = ["calculate", "solve", "find", "how many", "total", "cost"]
        has_math = any(indicator in question for indicator in math_indicators)
        
        return has_math and len(generated["question"]) >= 20 and len(generated["answer"]) >= 10
    
    def _quick_quality_check(self, generated: Dict) -> bool:
        """Quick quality check to filter out obviously bad responses before full assessment"""
        if not generated or not generated.get("question") or not generated.get("answer"):
            return False
        
        question = generated["question"].lower()
        answer = generated["answer"].lower()
        
        # Check for minimum content requirements
        if len(question) < 15 or len(answer) < 10:
            return False
        
        # Check for mathematical content indicators
        math_keywords = ["calculate", "solve", "find", "how many", "total", "cost", "price", "sum", "difference", "multiply", "divide", "add", "subtract"]
        has_math_keyword = any(keyword in question for keyword in math_keywords)
        
        # Check for numbers in the problem
        has_numbers = any(char.isdigit() for char in question)
        
        # Check for obvious non-math content
        bad_indicators = ["recipe", "story", "poem", "essay", "write a", "create a", "design", "explain why"]
        has_bad_content = any(indicator in question for indicator in bad_indicators)
        
        # Check for incomplete responses
        incomplete_indicators = ["...", "continue", "more", "etc", "and so on"]
        is_incomplete = any(indicator in answer for indicator in incomplete_indicators)
        
        return has_math_keyword and has_numbers and not has_bad_content and not is_incomplete
    
    def _assess_quality(self, generated: Dict) -> Dict:
        """Multi-dimensional quality assessment"""
        scores = {}
        
        # Basic validation
        question = generated.get("question", "").lower()
        answer = generated.get("answer", "").lower()
        
        # 1. Mathematical Content Assessment (0-1)
        math_indicators = ["calculate", "solve", "find", "how many", "total", "cost", "price", "sum", "difference"]
        math_operations = ["+", "-", "*", "/", "=", "×", "÷"]
        
        has_math_language = any(indicator in question for indicator in math_indicators)
        has_math_operations = any(op in answer for op in math_operations)
        has_numbers = any(char.isdigit() for char in (question + answer))
        
        scores["mathematical_content"] = (
            (0.4 if has_math_language else 0) +
            (0.3 if has_math_operations else 0) +
            (0.3 if has_numbers else 0)
        )
        
        # 2. Answer Quality Assessment (0-1)
        step_indicators = ["step", "first", "then", "next", "finally", "therefore"]
        has_reasoning = any(word in answer for word in step_indicators)
        has_final_answer = any(phrase in answer for phrase in ["answer is", "final answer", "result is", "equals"])
        answer_length_ok = 20 <= len(generated.get("answer", "")) <= 500
        
        scores["answer_quality"] = (
            (0.4 if has_reasoning else 0) +
            (0.3 if has_final_answer else 0) +
            (0.3 if answer_length_ok else 0)
        )
        
        # 3. Question Clarity Assessment (0-1)
        question_length_ok = 10 <= len(generated.get("question", "")) <= 200
        has_clear_ask = any(word in question for word in ["what", "how", "find", "calculate", "determine"])
        no_ambiguous_words = not any(word in question for word in ["maybe", "possibly", "might", "unclear"])
        
        scores["question_clarity"] = (
            (0.4 if question_length_ok else 0) +
            (0.3 if has_clear_ask else 0) +
            (0.3 if no_ambiguous_words else 0)
        )
        
        # 4. Confidence Score (from model if available)
        model_confidence = generated.get("confidence", 0.5)
        scores["model_confidence"] = min(max(model_confidence, 0), 1)
        
        # 5. Overall Quality Score (weighted average)
        weights = {
            "mathematical_content": 0.3,
            "answer_quality": 0.3,
            "question_clarity": 0.2,
            "model_confidence": 0.2
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights)
        scores["overall_quality"] = overall_score
        
        return scores


def _deduplicate_problems(problems: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
    """Remove near-duplicate problems based on question similarity"""
    if not problems:
        return problems
    
    # Simple deduplication based on question text similarity
    unique_problems = []
    seen_questions = []
    
    for problem in problems:
        question = problem.get("question", "").lower().strip()
        
        # Skip if too similar to existing questions
        is_duplicate = False
        for seen_q in seen_questions:
            # Simple similarity check - could be enhanced with more sophisticated methods
            similarity = _calculate_text_similarity(question, seen_q)
            if similarity > similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_problems.append(problem)
            seen_questions.append(question)
    
    print(f"Deduplication: {len(problems)} -> {len(unique_problems)} problems")
    return unique_problems


def _calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity based on word overlap"""
    if not text1 or not text2:
        return 0.0
    
    # Tokenize and normalize
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Synthetic Data Generation for Mathematical Word Problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode - generate few samples quickly
  python ray_sdg_job.py --test-mode
  
  # Production mode - generate full dataset
  python ray_sdg_job.py
  
  # Custom configuration
  python ray_sdg_job.py --test-mode --seeds 2 --variations 1
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
        "--workers",
        type=int,
        default=None,
        help="Number of workers to use (overrides auto-detection)"
    )
    
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=None,
        help="Quality threshold for filtering (0.0-1.0)"
    )
    
    return parser.parse_args()


def _get_shared_cache_directory() -> str:
    """Get shared cache directory for heavy data (models, datasets)"""
    # Priority order: shared PVC -> workspace PVC -> local fallback
    possible_cache_paths = [
        "/shared/cache",           # Shared PVC mount (highest priority)
        "/tmp/.cache"             # Local fallback (ephemeral)
    ]
    
    for cache_path in possible_cache_paths:
        try:
            os.makedirs(cache_path, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(cache_path, ".cache_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return cache_path
        except (OSError, PermissionError):
            continue
    
    # Fallback to /tmp if nothing else works
    return "/tmp/.cache"


def main():
    """Main function - supports both local testing and Ray cluster execution"""
    # Parse command line arguments
    args = parse_args()
    
    print("Starting distributed synthetic data generation...")
    print(f"Mode: {'TEST' if args.test_mode else 'PRODUCTION'}")
    
    # Check if we're running locally or in a Ray cluster
    try:
        import ray
        if not ray.is_initialized():
            # Local testing mode - initialize Ray locally
            print("Ray not initialized - starting local Ray instance for testing")
            ray.init()
            print("Local Ray instance started")
        
        print(f"Connected to Ray cluster: {ray.cluster_resources()}")
    except Exception as e:
        print(f"Error connecting to Ray cluster: {e}")
        return
    
    # Set up shared cache directory for dataset loading
    shared_cache_dir = _get_shared_cache_directory()
    print(f"Using shared cache directory: {shared_cache_dir}")
    
    # Set environment variables for HuggingFace datasets to use shared cache
    # Fix deprecation warning by using HF_HOME instead of TRANSFORMERS_CACHE
    os.environ['HF_HOME'] = shared_cache_dir
    os.environ['HF_DATASETS_CACHE'] = f"{shared_cache_dir}/datasets"
    
    # Load GSM8K dataset to shared storage
    print("Loading GSM8K dataset to shared storage...")
    gsm8k_dataset = load_dataset("gsm8k", "main", cache_dir=f"{shared_cache_dir}/datasets")
    print(f"Dataset loaded: {len(gsm8k_dataset['train'])} train samples, {len(gsm8k_dataset['test'])} test samples")
    
    # Configure based on mode and arguments
    if args.test_mode:
        # Test mode defaults
        default_seeds = 1
        default_variations = 1
        default_quality_threshold = 0.4
        print("🧪 TEST MODE: Using minimal samples for quick testing")
    else:
        # Production mode defaults
        default_seeds = 50
        default_variations = 3
        default_quality_threshold = 0.7
        print("🚀 PRODUCTION MODE: Generating full-scale dataset")
    
    # Apply user overrides or use defaults
    num_seed_samples = args.seeds if args.seeds is not None else default_seeds
    variations_per_seed = args.variations if args.variations is not None else default_variations
    quality_threshold = args.quality_threshold if args.quality_threshold is not None else default_quality_threshold
    
    print(f"Configuration:")
    print(f"  - Seed samples: {num_seed_samples}")
    print(f"  - Variations per seed: {variations_per_seed}")
    print(f"  - Quality threshold: {quality_threshold}")
    print(f"  - Expected total: {num_seed_samples} × {variations_per_seed} = {num_seed_samples * variations_per_seed} problems")
    print(f"Using {num_seed_samples} GSM8K problems as seeds")
    seed_samples = []
    
    # Get the first num_seed_samples from the train split
    train_data = gsm8k_dataset["train"]
    
    # Validate first sample structure
    if len(train_data) > 0:
        first_sample = train_data[0]
        print(f"Sample structure: {type(first_sample)}, keys: {list(first_sample.keys()) if isinstance(first_sample, dict) else 'Not a dict'}")
    
    for i in range(min(num_seed_samples, len(train_data))):
        sample = train_data[i]
        seed_samples.append({
            "seed_id": i,
            "question": sample["question"],
            "answer": sample["answer"]
        })
    
    print(f"Total seed samples prepared: {len(seed_samples)}")
    print(f"Expected total generations: {len(seed_samples)} seeds × {variations_per_seed} variations = {len(seed_samples) * variations_per_seed} problems")
    
    # Detect available resources and create appropriate workers
    cluster_resources = ray.cluster_resources()
    available_gpus = int(cluster_resources.get("GPU", 0))
    available_cpus = int(cluster_resources.get("CPU", 1))
    
    print(f"Cluster resources: {available_cpus} CPUs, {available_gpus} GPUs")
    
    # Determine optimal worker configuration
    use_gpu = available_gpus > 0
    if args.workers is not None:
        num_workers = args.workers
        print(f"Using user-specified worker count: {num_workers}")
    else:
        max_workers = min(available_gpus if use_gpu else available_cpus, len(seed_samples))
        num_workers = max(1, max_workers)  # At least 1 worker
        print(f"Auto-detected worker count: {num_workers}")
    
    print(f"Creating {num_workers} {'GPU' if use_gpu else 'CPU'} workers")
    
    # Create worker class with appropriate resources
    WorkerClass = create_worker_class(use_gpu=use_gpu)
    
    model_to_use = PRIMARY_MODEL
    config = MODEL_CONFIGS[model_to_use]
    
    print(f"Using model: {model_to_use} ({config['size']} params)")
    
    workers = []
    
    # Try to create workers with primary model, fallback if needed
    for i in range(num_workers):
        try:
            print(f"Creating worker {i+1}/{num_workers}...")
            worker = WorkerClass.remote(model_to_use, variations_per_seed)
            workers.append(worker)
            print(f"Worker {i+1} created successfully")
        except Exception as e:
            print(f"Failed to create worker {i+1} with {model_to_use}: {e}")
            
            # Try fallback models in order
            if model_to_use == PRIMARY_MODEL:
                print(f"Trying fallback model: {FALLBACK_MODEL}")
                try:
                    worker = WorkerClass.remote(FALLBACK_MODEL, variations_per_seed)
                    workers.append(worker)
                    model_to_use = FALLBACK_MODEL  # Switch to fallback for remaining workers
                    config = MODEL_CONFIGS[FALLBACK_MODEL]
                    print(f"Worker {i+1} created with fallback model")
                except Exception as fallback_e:
                    print(f"Fallback model also failed: {fallback_e}")
                    # Try backup model
                    print(f"Trying backup model: {BACKUP_MODEL}")
                    try:
                        worker = WorkerClass.remote(BACKUP_MODEL, variations_per_seed)
                        workers.append(worker)
                        model_to_use = BACKUP_MODEL
                        config = MODEL_CONFIGS[BACKUP_MODEL]
                        print(f"Worker {i+1} created with backup model")
                    except Exception as backup_e:
                        print(f"Backup model also failed: {backup_e}")
                        raise RuntimeError(f"Could not create worker with any available model")
            else:
                raise
    
    print(f"Created {len(workers)} Ray workers using {model_to_use} on {'GPU' if use_gpu else 'CPU'}")
    
    # Distribute work
    samples_per_worker = len(seed_samples) // num_workers
    futures = []
    
    print(f"Distributing {len(seed_samples)} seed samples across {num_workers} workers...")
    for i, worker in enumerate(workers):
        start_idx = i * samples_per_worker
        end_idx = start_idx + samples_per_worker if i < num_workers - 1 else len(seed_samples)
        worker_samples = seed_samples[start_idx:end_idx]
        
        print(f"Worker {i+1}: processing samples {start_idx}-{end_idx-1} ({len(worker_samples)} samples)")
        future = worker.generate_math_problems.remote(worker_samples)
        futures.append(future)
    
    # Collect results with progress tracking
    print("\nCollecting results from workers...")
    all_problems = []
    
    # Create a progress bar for worker completion
    with tqdm(total=num_workers, desc="Workers completed", unit="worker") as pbar:
        for i, future in enumerate(futures):
            worker_results = ray.get(future)
            all_problems.extend(worker_results)
            pbar.set_postfix({"Problems": len(all_problems)})
            pbar.update(1)
            print(f"Worker {i+1}: {len(worker_results)} problems generated")
    
    # Filter high-quality problems using multi-dimensional scoring
    print(f"\nFiltering {len(all_problems)} generated problems for quality...")
    # Use the configured quality threshold
    min_mathematical_content = max(0.4, quality_threshold - 0.2)
    min_answer_quality = max(0.3, quality_threshold - 0.3)
    
    high_quality_problems = []
    
    # Add progress bar for quality filtering
    with tqdm(all_problems, desc="Quality filtering", unit="problem") as pbar:
        for problem in pbar:
            quality_scores = problem.get("quality_scores", {})
            overall_quality = quality_scores.get("overall_quality", 0)
            math_content = quality_scores.get("mathematical_content", 0)
            answer_quality = quality_scores.get("answer_quality", 0)
            
            # Multi-criteria filtering
            if (overall_quality >= quality_threshold and 
                math_content >= min_mathematical_content and 
                answer_quality >= min_answer_quality):
                high_quality_problems.append(problem)
            
            pbar.set_postfix({"High quality": len(high_quality_problems)})
    
    print(f"Quality filtering complete: {len(high_quality_problems)}/{len(all_problems)} problems passed")
    
    # Remove near-duplicates based on question similarity
    print("Removing duplicate problems...")
    high_quality_problems = _deduplicate_problems(high_quality_problems)
    
    # Print comprehensive statistics
    print(f"\n{'='*60}")
    print(f"SYNTHETIC DATA GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total problems generated: {len(all_problems)}")
    print(f"High quality problems: {len(high_quality_problems)}")
    print(f"Quality pass rate: {len(high_quality_problems)/len(all_problems)*100:.1f}%" if all_problems else "0.0%")
    
    if high_quality_problems:
        # Calculate average quality scores
        avg_overall = sum(p["overall_quality"] for p in high_quality_problems) / len(high_quality_problems)
        avg_math = sum(p["quality_scores"]["mathematical_content"] for p in high_quality_problems) / len(high_quality_problems)
        avg_answer = sum(p["quality_scores"]["answer_quality"] for p in high_quality_problems) / len(high_quality_problems)
        
        print(f"Average quality scores:")
        print(f"  Overall: {avg_overall:.3f}")
        print(f"  Mathematical content: {avg_math:.3f}")
        print(f"  Answer quality: {avg_answer:.3f}")
    
    print(f"{'='*60}")
    
    # Save dataset
    train_size = int(0.8 * len(high_quality_problems))
    # Calculate quality statistics
    if high_quality_problems:
        avg_overall_quality = sum(p["overall_quality"] for p in high_quality_problems) / len(high_quality_problems)
        difficulty_distribution = {}
        for p in high_quality_problems:
            diff = p.get("difficulty", "unknown")
            difficulty_distribution[diff] = difficulty_distribution.get(diff, 0) + 1
    else:
        avg_overall_quality = 0
        difficulty_distribution = {}
    
    synthetic_dataset = {
        "train": high_quality_problems[:train_size],
        "test": high_quality_problems[train_size:],
        "metadata": {
            "total_generated": len(all_problems),
            "high_quality_count": len(high_quality_problems),
            "quality_threshold": quality_threshold,
            "min_mathematical_content": min_mathematical_content,
            "min_answer_quality": min_answer_quality,
            "avg_overall_quality": round(avg_overall_quality, 3),
            "difficulty_distribution": difficulty_distribution,
            "model_used": model_to_use,
            "generation_method": "ray_distributed_qwen",
            "features": [
                "structured_json_output",
                "multi_dimensional_quality_assessment", 
                "difficulty_variation",
                "deduplication",
                "robust_parsing"
            ]
        }
    }
    
    # Save to shared persistent storage
    # Priority order: shared PVC -> workspace PVC -> local fallback
    possible_paths = [
        "/shared/datasets",         # Shared PVC mount (highest priority)
        "/tmp/synthetic_data"      # Local fallback (ephemeral)
    ]
    
    output_path = None
    for path in possible_paths:
        try:
            os.makedirs(path, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(path, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            output_path = path
            print(f"Using storage path: {output_path}")
            break
        except (OSError, PermissionError):
            continue
    
    if not output_path:
        raise RuntimeError("No writable storage path found!")
    
    dataset_file = f"{output_path}/synthetic_dataset.json"
    with open(dataset_file, "w") as f:
        json.dump(synthetic_dataset, f, indent=2)
    
    print(f"Dataset saved: {len(synthetic_dataset['train'])} train / {len(synthetic_dataset['test'])} test")
    print(f"Saved to: {dataset_file}")
    
    # Also save metadata for debugging
    metadata_file = f"{output_path}/dataset_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(synthetic_dataset["metadata"], f, indent=2)
    print(f"Metadata saved to: {metadata_file}")


if __name__ == "__main__":
    main()
