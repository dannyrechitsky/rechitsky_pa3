#!/bin/bash
#SBATCH --job-name=cos_231
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

# Output/error for debugging
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log

# Notifications for when the job ends or fails
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dannyrechitsky@brandeis.edu

# Make sure cached tokenizer used (Shel's implementation 12/17)
export HF_HOME=/data/dannyrechitsky/.cache/huggingface
export UV_CACHE_DIR=/data/dannyrechitsky/.cache/uv

# (optional)
# export <ENV_VAR> # (optional)

# source uv env
source .venv/bin/activate

# TASK 1 ENCODER - RoBERTa QA ###

# run baseline pretrained model
# uv run main.py --model_type pretrained

# run different training sizes (30%, 50% and 100%) with fixed LoRA rank 8
uv run main.py --train_pct 0.3 --lora_rank 8 --epochs 3 --load_model_path latest 
uv run main.py --train_pct 0.5 --lora_rank 8 --epochs 3 --load_model_path latest
# uv run main.py --train_pct 1.0 --lora_rank 8 --epochs 3 --load_model_path latest

# run different LoRA ranks with 100% training data (=best results from above runs)
# uv run main.py --train_pct 1.0 --lora_rank 8 --epochs 3 --load_model_path latest
# uv run main.py --train_pct 1.0 --lora_rank 16 --epochs 3 --load_model_path latest 
# uv run main.py --train_pct 1.0 --lora_rank 32 --epochs 3 --load_model_path latest 

# NOTE current full training steps --train_steps 9799 (2+ epochs)

