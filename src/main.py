import os
from datasets import load_dataset
from dataset import get_tokenizer, preprocess_data, get_train_val_test_set
import json
from functools import partial
from transformers import \
    DataCollatorWithPadding, \
    TrainingArguments, \
    AutoModelForQuestionAnswering, \
    Trainer
import evaluate
import numpy as np
import argparse
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model

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
    
    # 4. Load Model Path (The consolidated argument)
    # This argument is highly flexible:
    #   - If provided:Loads the model from the specified path.
    #   - If set to 'latest': Triggers the search for the highest checkpoint in the current output_dir.
    #   - If None (default): Starts with the RoBERTa-base model.
    parser.add_argument("--load_model_path", type=str, default=None, 
                        help="Path to a specific saved checkpoint \
                            (e.g., 'results/rank=8_size=0.5') or set \
                                to 'latest' to automatically find the \
                                highest checkpoint in the current \
                                output directory."
    )

    # 5. Train Steps
    parser.add_argument("--train_steps", type=int, default=-1,
                    help="The maximum number of training steps \
                        to run. Set to -1 to use epochs instead or \
                            only perform evaluation."
    )

    # 6. Epochs
    parser.add_argument("--epochs", type=int, default=3,
                    help="Number of full passes over the training data. \
                    only if num_train_epochs=args.epochs in training_args"
                    )
    

    return parser.parse_args()

if __name__ == "__main__":
    #### DELETE LATER: just to print sample for reference #####
    # raw_datasets = load_dataset("rajpurkar/squad_v2")
    # train_data = raw_datasets['train']
    
    # with open("sample.json", "w") as f:
    #     json.dump(train_data[0], f, indent=4, ensure_ascii=False)

    
    ### ARGPARSE FOR SHELL SCRIPT ###
    args = parse_args()
    
    # Now you can use these variables anywhere!
    print(f"Running experiment with: {args.model_type}, Data: {args.train_pct}, Rank: {args.lora_rank}")
    

    ### DATA PREPROCESSING ###

    raw_datasets = load_dataset("rajpurkar/squad_v2")
    tokenizer = get_tokenizer()

    preprocess_function = partial(preprocess_data, 
                                  tokenizer=tokenizer)

    # remove all raw dataset columns, the just get in the way
    cols_to_remove = ["id", "title", "context", "question", "answers"]
    
    # tokenize datasets
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True, 
        remove_columns=cols_to_remove
    )

    # split datasets into training, validation and testing
    train_set, val_set, test_set = get_train_val_test_set(
        tokenized_datasets
    )
    
    # reduce train set to 30% or 50% based on argparse
    if args.train_pct < 1.0:
    # select the first N% of the training data
        max_train_samples = int(len(train_set) * args.train_pct)
        train_set = train_set.select(range(max_train_samples))



    ### TRAINING ###

    # initialize data collator
    data_collator = DataCollatorWithPadding(tokenizer)

    # set output directory
    if args.model_type=='finetuned':
        exp_name = f"train={args.train_pct}_rank={args.lora_rank}"
    else: # pretrained
        exp_name = f"pretrained"
    output_dir = os.path.join("results", exp_name)

    #---Load Model---#

    # 1. Start with the default pretrained model
    model_to_load = "roberta-base"
    path_to_load = None

    # 2. Check if the user specified a model path via argparse (highest priority)
    if args.load_model_path is not None \
    and args.load_model_path != "latest":
        path_to_load = args.load_model_path
        
    # 3. Check if the user wants to load the latest checkpoint from the *current* experiment folder
    # and that the folder exists
    elif args.load_model_path == "latest" and os.path.isdir(output_dir):
        # Use the experiment's output directory to find the latest checkpoint
        latest_checkpoint = get_last_checkpoint(output_dir)
        
        # If a checkpoint is found, set the path to load it
        if latest_checkpoint is not None:
            path_to_load = latest_checkpoint
            print(f"Loading latest checkpoint found in {output_dir}: {path_to_load}")
        else:
            # Fallback if the folder exists but is empty
            print(f"Warning: latest model specified but no checkpoints found in {output_dir}. \
                Starting from scratch.")
            path_to_load = model_to_load # Use roberta-base as a fallback

    # 4. If no specific path was set, use the default roberta-base
    else:
        path_to_load = model_to_load
    

    # Final Model Loading Action
    model = AutoModelForQuestionAnswering.from_pretrained(path_to_load)


    #---Configure LoRA---#

    if args.lora_rank > 0:
        # Set up LoRA parameters
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=32,
            target_modules=["query", "key", "value", "output.dense"], 
            lora_dropout=0.1,
            task_type="QUESTION_ANS"
        )
    
        # Convert model to LoRa
        model = get_peft_model(model, config)

        # Print num trainable parameters
        model.print_trainable_parameters()


    #---Configure Trainer---#

    # set training arguments
    max_steps = args.train_steps

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        num_train_epochs=args.epochs,
        save_steps=500,
        logging_steps=50, # to see clear loss curve
        save_total_limit=2, # save space by deleting older checkpoints
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        dataloader_num_workers=4
    )

    # initialize trainer with args
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_set,
        eval_dataset=val_set,
        processing_class=tokenizer
    )

    # Only train if model_type is explicitly set to 'finetuned'
    if args.model_type == "finetuned":
        print("Model type 'finetuned' selected. Starting training...")
        
        if path_to_load and os.path.isdir(path_to_load):
            print(f"Resuming trainer state from: {path_to_load}")
            trainer.train(resume_from_checkpoint=path_to_load)
        else:
            trainer.train()
            
    else:
        # If model_type is 'pretrained' (default), skip training
        print(f"Model type '{args.model_type}' selected. Skipping training to evaluate baseline.")
    

    
    ### EVALUATE VAL SET ###

    # get predictions_output: (predictions, label_ids, metrics)
    predictions_output = trainer.predict(val_set)

    # extract start and end logits from predictions
    start_logits_all, end_logits_all = \
        predictions_output.predictions

    # initialize list of prediction dictionaries
    predictions = []

    # initialize list of refences (truth) to compare with predictions
    references = []

    # --------- best f1 threshold so far ------------ #
    # used earlier in validation for calibrating no_answer probability
    # null_threshold = 6.25

    ## iterate over val set to compare prediction|actual
    for i in range(len(val_set)):
        
        # assign start and end logits
        start_logits = start_logits_all[i]
        end_logits = end_logits_all[i]

        # calculate no_answer_score: prob that there is no answer
        no_answer_score = start_logits[0] + end_logits[0]

        # find best "has answer" span (ignoring 0)
        start_logits[0] = -9999 # remove "no answer" as possibility
        end_logits[0] = -9999 # remove "no answer" as possibility
        start_index = np.argmax(start_logits)
        end_index = np.argmax(end_logits)

        if start_index > end_index:
            # Fallback: if model is confused, just predict the single start token
            end_index = start_index

        # get real answer(s) for this sample
        real_answers = val_set[i]['answers']

        ## build predictions dictionaries
        # get unique question ID to use in evaluation metric
        qid = val_set[i]['example_id']

        # calculate score for best span
        start_index = np.argmax(start_logits)
        end_index = np.argmax(end_logits)
        span_score = start_logits[start_index] + end_logits[end_index]

        # decode and predict
        pred_answer = tokenizer.decode(
                val_set[i]["input_ids"][start_index : end_index + 1],
                skip_special_tokens=True # remove CLS token if it sneaks in
            )
        
        # calculate score difference to predict answer vs no answer below
        score_diff = no_answer_score - span_score

        # predictions: add qid|answer pair as dictionary to predictions
        predictions.append({
            'id': qid, 
            'prediction_text': pred_answer,
            'no_answer_probability': float(score_diff),

        })

        # truth: add qid|answer pair as dictionary to references
        references.append({'id': qid, 
                           'answers' : real_answers
        })

    

    # initialize metric
    metric = evaluate.load("squad_v2")

    # compute results
    val_results = metric.compute(predictions=predictions, 
                             references=references
    )

    # --- Save Evaluation Results ---
    results_filename = os.path.join(output_dir, "evaluation_results.json")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    with open(results_filename, "w") as f:
        json.dump(val_results, f, indent=4)
        
    print("\n--- Final Evaluation Results ---")
    print(json.dumps(val_results, indent=4))
    print(f"Results saved to: {results_filename}")
    

        
     
    ### TEST SET INFERENCE ###

    # 1. Capture the threshold programmatically (No manual entry needed!)
    optimal_threshold = val_results.get("best_f1_thresh", 0.0)
    
    print(f"\n Calibration Complete.")
    print(f"Optimal Threshold Captured: {optimal_threshold}")
    print(f"Propagating this threshold to Test Phase...")

    print("\n--- Phase 2: Running Test Set with Threshold ---")

    # 2. Get Raw Test Predictions
    test_predictions_output = trainer.predict(test_set)
    t_start_logits_all, t_end_logits_all = test_predictions_output.predictions

    test_final_preds = []
    test_refs = []

    for i in range(len(test_set)):
        
        # --- A. Calculate Raw Scores (Same as Phase 1) ---
        start_logits = t_start_logits_all[i]
        end_logits = t_end_logits_all[i]
        
        # Calculate No Answer Score ([CLS] token)
        no_answer_score = start_logits[0] + end_logits[0]

        # Calculate Best Span Score (masking [CLS] to find best text)
        start_logits_candidate = start_logits.copy()
        end_logits_candidate = end_logits.copy()
        start_logits_candidate[0] = -9999 
        end_logits_candidate[0] = -9999
        
        start_index = np.argmax(start_logits_candidate)
        end_index = np.argmax(end_logits_candidate)

        # Sanity check: if end < start, force them to match
        if start_index > end_index:
            end_index = start_index

        span_score = start_logits[start_index] + end_logits[end_index]
        
        # Decode the raw candidate text
        pred_text = tokenizer.decode(
            test_set[i]["input_ids"][start_index : end_index + 1],
            skip_special_tokens=True
        )

        # --- B. Apply The Decision Boundary ---
        score_diff = no_answer_score - span_score

        # Use the variable 'optimal_threshold' we captured above
        if score_diff > optimal_threshold:
            final_answer = "" # The gap is too large -> Predict No Answer
        else:
            final_answer = pred_text # The gap is small enough -> Predict Text

        # --- C. Store Final Decision ---
        # Note: We hardcode 'no_answer_probability' to 0.0 because 
        # we have already made the hard decision ourselves.
        test_final_preds.append({
            'id': test_set[i]['example_id'], 
            'prediction_text': final_answer,
            'no_answer_probability': 0.0 
        })
        
        # Collect references
        if 'answers' in test_set[i]:
            test_refs.append({
                'id': test_set[i]['example_id'], 
                'answers': test_set[i]['answers']
            })

    # 3. Compute Final Test Scores (using our fixed decision)
    test_results = metric.compute(predictions=test_final_preds, references=test_refs)
    print("\n--- Final Test Set Results (Applied Threshold) ---")
    print(json.dumps(test_results, indent=4))
    
    # 4. Save Final Predictions to JSON
    # We save the full list of predictions so you can inspect them manually
    results_filename = os.path.join(output_dir, "test_predictions_final.json")
    
    # # Create a simplified output format (ID + Answer)
    # final_output_structure = {
    #     "metrics": test_results,
    #     "threshold_used": optimal_threshold,
    #     "predictions": test_final_preds
    # }

    with open(results_filename, "w") as f:
        json.dump(test_results, f, indent=4)

    print(f"Final predictions saved to: {results_filename}")



    ### SAVE LOSS CURVES FOR TRAINING AND VALIDATION ###
    
    # Extract the history from the trainer
    history = trainer.state.log_history

    # Organize data into two clean lists
    train_loss = []
    val_loss = []

    for entry in history:
        if "loss" in entry: # This is a training step
            train_loss.append({
                "step": entry["step"], 
                "loss": entry["loss"]
            })
        elif "eval_loss" in entry: # This is a validation step
            val_loss.append({
                "step": entry["step"], 
                "loss": entry["eval_loss"]
            })

    # Save to file
    loss_file = os.path.join(output_dir, "loss_curves.json")
    with open(loss_file, "w") as f:
        json.dump({"train": train_loss, "val": val_loss}, f, indent=4)
        
    print(f"Loss curves saved to: {loss_file}")
    
    