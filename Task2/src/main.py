import os
from datasets import load_dataset
from dataset import *
import numpy as np
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
import torch
import argparse
from transformers.trainer_utils import get_last_checkpoint

import json
import re
from sacrebleu.metrics import BLEU

import subprocess
import tempfile

from transformers import AutoModelForCausalLM
import sys

import resource
import concurrent.futures
import multiprocessing



def print_gpu_utilization():
    """Print GPU utilization"""
    if torch.cuda.is_available():
        # Shows how much memory is currently used versus the total available
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"--------------------------------------------"
            f"GPU Memory: {used:.2f}GB / {total:.2f}GB used")


def parse_args():
    """Parse CLI arguments"""
    parser = argparse.ArgumentParser(description="Fine-tuning configuration")
    
    # 1. Trainset Size (float, e.g., 0.3 for 30%)
    parser.add_argument("--train_pct", type=float, default=1.0, 
                        help="Fraction of training data to use (0.0 - 1.0)")
    
    # 2. LoRA Rank (int, e.g. 8, 16, 32)
    parser.add_argument("--lora_rank", type=int, default=0, 
                        help="Rank dimension for LoRA. If set to 0, no LoRA")
    
    # 3. Model Type (string choice)
    parser.add_argument("--model_type", type=str, choices=["pretrained", "finetuned"], 
                        default="finetuned", help="Which model version to start with")
    
    # 4. Load Model Path
    parser.add_argument("--load_model_path", type=str, default=None, 
                        help="Path to a specific saved checkpoint \
                            (e.g., 'results/rank=8_size=0.5') or set \
                                to 'latest' to automatically find the \
                                highest checkpoint in the current \
                                output directory. Or set to None \
                                    for base model"
    )

    # 5. Epochs
    parser.add_argument("--epochs", type=int, default=3,
                    help="Number of full passes over the training data.")
    
    # 6. Eval From File Only
    parser.add_argument("--eval_from_file_only", type=str, default=None,
                        help="Path to an existing .json file to skip " \
                        "generation and run evaluation metrics only.")
    
    
    return parser.parse_args()

    
# This code will be injected at the TOP of every generated script.
# It overrides built-in functions to block absolute paths and parent directories.
SAFETY_HEADER = """
import builtins
import os

def _audit_path(path):
    # Allow integer file descriptors (standard pipes)
    if isinstance(path, int): return
    
    s_path = str(path)
    
    # 1. Block Absolute Paths (e.g., /home/user/...)
    if os.path.isabs(s_path):
        raise PermissionError(f"SANDBOX VIOLATION: Absolute paths are forbidden: {s_path}")
        
    # 2. Block Parent Directory traversal (e.g., ../../)
    if ".." in s_path:
        raise PermissionError(f"SANDBOX VIOLATION: Parent directory references are forbidden: {s_path}")

# Override builtins.open
# We allows reads (for imports to work) but strictly audit WRITES.
_orig_open = builtins.open
def _safe_open(file, mode='r', *args, **kwargs):
    # If mode implies writing ('w', 'a', 'x', '+'), enforce checks
    if any(m in mode for m in ['w', 'a', 'x', '+']):
        _audit_path(file)
    return _orig_open(file, mode, *args, **kwargs)

builtins.open = _safe_open

# Override destructive os functions
# These should NEVER use absolute paths in a sandbox.
def _wrap_os_func(func_name):
    if hasattr(os, func_name):
        _orig_func = getattr(os, func_name)
        def _safe_func(path, *args, **kwargs):
            _audit_path(path)
            return _orig_func(path, *args, **kwargs)
        setattr(os, func_name, _safe_func)

# Block common file system modification commands
for fn in ['remove', 'rmdir', 'mkdir', 'makedirs', 'rename', 'replace', 'unlink']:
    _wrap_os_func(fn)

# ---------------------------------------------------------
"""

def set_limits():
    """Sets resource limits to prevent OOM or disk filling."""
    resource.setrlimit(resource.RLIMIT_CPU, (2, 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))

def run_code_safely(code_str, time_limit=2):
    """
    Executes Python code in a sandboxed environment with path auditing.
    """
    # 1. Sanitize Environment (Remove Secrets)
    safe_env = os.environ.copy()
    keys_to_remove = ['HF_TOKEN', 'WANDB_API_KEY', 'OPENAI_API_KEY', 'HUGGING_FACE_HUB_TOKEN']
    for key in keys_to_remove:
        safe_env.pop(key, None)

    # 2. Create Temp Directory Sandbox
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, 'script.py')
        
        try:
            # 3. Inject Safety Header + User Code
            full_code = SAFETY_HEADER + "\n" + code_str
            
            with open(script_path, 'w') as f:
                f.write(full_code)

            # 4. Run Subprocess
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=temp_dir,             # SANDBOX: Traps relative paths
                env=safe_env,             # PRIVACY: Hides tokens
                stdin=subprocess.DEVNULL, # ANTI-HANG: Blocks input()
                capture_output=True,
                text=True,
                timeout=time_limit,
                preexec_fn=set_limits     # RESOURCES: Prevents disk/mem overflow
            )
            
            success = (result.returncode == 0)
            return result.stdout.strip(), result.stderr.strip(), success

        except subprocess.TimeoutExpired:
            return "", "TIMEOUT: Execution exceeded time limit", False
        except Exception as e:
            return "", f"SYSTEM ERROR: {str(e)}", False

def clean_code(code_str):
    """
    Extracts code from Markdown blocks. 
    Handles 'Here is the code:' intros and explanatory outros.
    """
    if not code_str:
        return ""

    # 1. Try to find a block starting with ```python or just ```
    #    re.DOTALL allows the (.) to match newlines (crucial for multi-line code)
    match = re.search(r"```(?:python)?\n?(.*?)```", code_str, re.DOTALL)
    
    if match:
        # We found a block! Return just the content inside.
        return match.group(1).strip()
    
    # 2. Edge Case: Sometimes models output code with indentation but no backticks.
    #    If no markdown block is found, return the original string.
    #    (If it was pure text like "Yes it is prime", this will fail execution, which is correct).
    return code_str.strip()



import concurrent.futures

# --- HELPER FOR PARALLEL EXECUTION ---
def evaluate_single_item(item):
    """
    Independent function to evaluate one item.
    Must be defined at the top level (outside other functions) for multiprocessing to work.
    """
    # 1. Run Predicted Code
    pred_out, pred_err, pred_success = run_code_safely(item['predicted_code'])
    
    # 2. Run Ground Truth Code
    true_out, true_err, true_success = run_code_safely(item['true_code'])
    
    # 3. Determine Status
    # Default: Not a match, Ground Truth not valid
    is_syntax_pass = pred_success
    is_gt_executable = False
    is_exec_match = False

    # Check if Ground Truth is valid (Success AND produces output)
    if true_success and (true_out or true_err):
        is_gt_executable = True
        
        # Check for Match
        if pred_success and (pred_out == true_out) and (pred_err == true_err):
            is_exec_match = True
            
    return is_syntax_pass, is_gt_executable, is_exec_match

# --- MAIN EVALUATION FUNCTION ---
def eval_from_file(input_file):
    print(f"Loading predictions from: {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Could not find file: {input_file}")

    # 1. Load Data
    with open(input_file, "r") as f:
        data = json.load(f)
        
    # 2. BLEU Score (Fast enough to keep single-threaded)
    print("Calculating BLEU score...")
    bleu = BLEU()
    refs = [[item["true_code"] for item in data]]
    preds = [item["predicted_code"] for item in data]
    score = bleu.corpus_score(preds, refs)
    print(f"BLEU Score: {score.score:.2f}")

    # 3. Execution Score (PARALLELIZED)
    print(f"Running execution-based evaluation on {len(data)} samples...")
    print("Spawning processes... (This may take a moment)")
    
    ctx = multiprocessing.get_context("spawn") 
    with concurrent.futures.ProcessPoolExecutor(mp_context=ctx) as executor:
        results = list(executor.map(evaluate_single_item, data))

    # 4. Aggregate Results
    # results is a list of tuples: (is_syntax_pass, is_gt_executable, is_exec_match)
    
    total_syntax_passes = sum(1 for r in results if r[0])
    total_executable = sum(1 for r in results if r[1])
    execution_matches = sum(1 for r in results if r[2])

    # 5. Calculate Metrics
    exec_accuracy = 0.0
    if total_executable > 0:
        exec_accuracy = (execution_matches / total_executable) * 100
        
    syntax_pass_rate = 0.0
    if len(data) > 0:
        syntax_pass_rate = (total_syntax_passes / len(data)) * 100

    print(f"Execution Accuracy: {exec_accuracy:.2f}%")
    print(f"Syntax Pass Rate:   {syntax_pass_rate:.2f}%")
    print(f"Valid Samples (GT produced output): {total_executable}/{len(data)}")

    # 6. Save Metrics
    final_metrics = {
        "BLEU": score.score,
        "execution_accuracy": exec_accuracy,
        "syntax_pass_rate": syntax_pass_rate,
        "total_samples": len(data),
        "executable_samples": total_executable,
        "syntax_passes": total_syntax_passes,
        "execution_matches": execution_matches
    }
    
    # Derive output filename
    input_dir = os.path.dirname(input_file)
    metrics_file = os.path.join(input_dir, "metrics_recalc.json")
    
    with open(metrics_file, "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"Metrics saved to: {metrics_file}")
    
    sys.exit(0)


if __name__ == "__main__":

    ### PARSE CLI ARGS ###
    args = parse_args()

    # SHORT CIRCUIT: Eval From File
    if args.eval_from_file_only:
        eval_from_file(args.eval_from_file_only)

    ### PREPROCESS DATA ###

    # load and split datasets
    raw_data = load_dataset('flytech/python-codes-25k')
    train_set, val_set, test_set = get_train_val_test_set(raw_data)

    # reduce train set to 30% or 50% based on argparse
    if args.train_pct < 1.0:
    # select the first N% of the training data
        max_train_samples = int(len(train_set) * args.train_pct)
        train_set = train_set.select(range(max_train_samples))

    # initialize tokenizer
    tokenizer = get_tokenizer()
    tokenizer.padding_side = "left" # for batch processing

    # ensure a pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id



    ### SET UP LoRA ###

    # LoRA applied
    if args.lora_rank > 0:
        lora_config = LoraConfig(
            r=args.lora_rank,                           # Rank: How many new parameters to add
            lora_alpha=32,                  # Scaling (usually 2x Rank)
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Targets Attention
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        learning_rate = 2e-4 # standard LoRA rate

    # LoRA NOT applied
    else:
        lora_config = None
        learning_rate = 5e-6 # much lower to prevent model collapse w/o LoRA




    ### TRAIN ###
    # Load Model Manually (Fixes the memory leak)
    # use bfloat16 to match hardware's capability
    print("Loading model manually...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CHECKPOINT,
        torch_dtype=torch.bfloat16,  # Standard half-precision (efficient & accurate)
        device_map="auto"            # Let Accelerate handle placement safely
    )


    # fine-tuning config
    sft_config = SFTConfig(
        dataset_text_field="text",
        packing=False,
        max_length=512,
        
        # Critical for Full Fine-Tuning on 1 GPU
        gradient_checkpointing=True,    # Frees massive VRAM
        bf16=True,                      # Stability
        optim="adamw_bnb_8bit",         # Optimizer memory reduction
        
        per_device_train_batch_size=1,  # Start at 1 to be safe; increase if VRAM allows
        gradient_accumulation_steps=16, # High accumulation to keep effective batch size 16
        
        num_train_epochs=args.epochs,             # Scales with 3 training sizes
        learning_rate=learning_rate,             
        save_strategy="epoch",
        eval_strategy="epoch"
    )

    # Create a unique folder name based on your experiment settings
    if args.model_type=='finetuned':
        exp_name = f"train={args.train_pct}_rank={args.lora_rank}"
    else: # pretrained
        exp_name = f"pretrained"
    sft_config.output_dir = os.path.join("./results", exp_name)


    # initialize trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_set,
        eval_dataset=val_set,
        peft_config=lora_config,
        formatting_func=get_formatting_function(tokenizer),
        args=sft_config
    )

    # check for initial spike 90% memory
    # that signifies imminent OOM
    print_gpu_utilization()


    # Logic to determine checkpoint
    resume_from_checkpoint = None

    # Case 1: Auto-resume ("latest")
    if args.load_model_path == "latest":
        if os.path.isdir(sft_config.output_dir):
            resume_from_checkpoint = get_last_checkpoint(sft_config.output_dir)
            if resume_from_checkpoint:
                print(f"Auto-detected latest checkpoint: {resume_from_checkpoint}")
            else:
                print(f"No checkpoint found in {sft_config.output_dir}. Starting fresh.")

    # Case 2: Explicit path provided
    elif args.load_model_path is not None:
        if os.path.isdir(args.load_model_path):
            resume_from_checkpoint = args.load_model_path
            print(f"Loading specific checkpoint: {resume_from_checkpoint}")
        else:
            # CRITICAL: Crash if the user-requested path is missing
            raise FileNotFoundError(f"Checkpoint not found at: {args.load_model_path}")


    # Train only if we are not just testing the pretrained model
    if args.model_type == "pretrained":
        print("Skipping training (Model Type: Pretrained)...")
    elif resume_from_checkpoint is not None:
        print(f"Resuming training from {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        print("Starting fresh training run...")
        trainer.train()


    ### EVALUATE ###

    ## 1. Generate predicted python code in BATCHES
    BATCH_SIZE = 16  # Start with 16. If you get OOM (Out of Memory), lower to 8 or 4.
    print(f"Starting Generation for Evaluation (Batch Size: {BATCH_SIZE})...", flush=True)

    # Ensure output directory exists before saving
    os.makedirs(sft_config.output_dir, exist_ok=True)

    results = []
    output_file = os.path.join(sft_config.output_dir, "test_predictions.json")

    # Ensure model is in eval mode and grab the correct model object
    if hasattr(trainer, "model_wrapped"):
        model = trainer.model_wrapped     
    else: 
        model = trainer.model
    model.eval()

    # Access data columns directly for easier slicing
    all_instructions = test_set['instruction']
    all_true_codes = test_set['output']
    total_samples = len(all_instructions)

    # Loop through data in chunks of BATCH_SIZE
    
    
    for i in range(0, total_samples, BATCH_SIZE):
    # for i in range(0, 32, BATCH_SIZE): # UNCOMMENT for SMALL TEST SET
        # 1. Slice the batch
        batch_instructions = all_instructions[i : i + BATCH_SIZE]
        batch_true_codes = all_true_codes[i : i + BATCH_SIZE]

        # 2. Prepare Inputs
        # Create prompt strings for the whole batch
        batch_prompts = [
            f"### Instruction:\n{inst}\n\n### Response:\n" for inst in batch_instructions
        ]
        
        # Tokenize batch (padding=True is CRITICAL here so tensors match size)
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(model.device)
        
        # 3. Generate for Batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                pad_token_id=tokenizer.eos_token_id
            )

        # 4. Decode Batch
        full_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # 5. Process Batch Results
        for j, full_output in enumerate(full_outputs):
            # Extract response part
            try:
                generated_code = full_output.split("### Response:\n")[1].strip()
            except IndexError:
                generated_code = full_output

            # Clean codes
            cleaned_pred_code = clean_code(generated_code)
            cleaned_true_code = clean_code(batch_true_codes[j])

            results.append({
                "instruction": batch_instructions[j],
                "true_code": cleaned_true_code,
                "predicted_code": cleaned_pred_code
            })

        # 6. Save Progress (Less frequently to save IO time)
        # Save every 10 batches (e.g., every 160 items)
        current_count = min(i + BATCH_SIZE, total_samples)
        if (i // BATCH_SIZE) % 10 == 0 or current_count == total_samples:
            print(f"Processed {current_count}/{total_samples} samples...", flush=True)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)

    
    ## 2. store results
    # export to JSON
    output_file = os.path.join(sft_config.output_dir, "test_predictions.json")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Saved {len(results)} predictions to {output_file}")

    

    ## 3. evaluate
    # Use the helper function we defined earlier!
    # Note: This function calculates BLEU + Execution and saves to JSON, then exits.
    print("Running final evaluation...")
    eval_from_file(output_file)