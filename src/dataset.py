from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict

MODEL_CHECKPOINT = "roberta-base"

def get_tokenizer():
    """loads and returns RoBERTa tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    return tokenizer

def preprocess_data(examples, tokenizer):
    """tokenize the questions and context"""
    questions = list(examples["question"])
    contexts = list(examples["context"])



    model_inputs = tokenizer(
        questions, 
        contexts, 
        truncation=True,
        stride=128,
        return_offsets_mapping=True, 
        padding="max_length",
        max_length=512,
        return_overflowing_tokens=True
    )

    # Initialize a list for the expanded IDs
    sample_ids = []
    expanded_answers = []

    offset_mapping = model_inputs["offset_mapping"]

    # extract start and end token indices over all the examples
    start_token_indices = []
    end_token_indices = []

    for i, (input_ids, offset) in \
    enumerate(zip(model_inputs["input_ids"], model_inputs["offset_mapping"])):
        # This ID links the current tokenized sequence (i) back to the original example.
        # **This avoids long samples being cut off between batches in the map function** 
        # The defensive fix for mapping the original example index (0-999)
        if "overflow_to_sample_mapping" in model_inputs:
            original_example_index = model_inputs["overflow_to_sample_mapping"][i]
        else:
            original_example_index = i

        # The defensive fix for mapping the original example index (0-999)
        # Expand the original ID list
        # We use the corrected index to find the ID from the original batch.
        original_id = examples["id"][original_example_index]
        sample_ids.append(original_id)

        original_answer_structure = examples["answers"][original_example_index]
        expanded_answers.append(original_answer_structure)

        # 1. get the sequence IDs to find the context
        sequence_ids = model_inputs.sequence_ids(i)

        # 2. get start token index and end token index for 
        # this example's context
        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1

        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1
        
        # 3. get answer char indices
        answer = examples["answers"][original_example_index]

        # 3a. check for empty answers, label them as out of context
        if len(answer["answer_start"]) == 0:
            start_token_indices.append(0)
            end_token_indices.append(0)
            continue

        # 3b. answer not empty, get char indices
        start_char = answer["answer_start"][0]
        answer_text = answer["text"][0]
        end_char = start_char + len(answer_text) - 1

        # 4. check if answer outside of context
        if start_char < offset[context_start][0] \
        or end_char > offset[context_end][1]:
            # 5 if answer out of context, add [CLS] token index
                start_token_indices.append(0)
                end_token_indices.append(0)
        else:
            # 6. else find the specific token index where
            # the answer starts and ends
            
            # loop over offset until answer start char found
            idx_start = context_start
            while start_char >= offset[idx_start][1]:
                idx_start += 1
            
            # add token start index to list
            start_token_indices.append(idx_start)

            # loop backwards over offset until answer end char found
            idx_end = context_end
            # FIX: Add a safeguard to stop the loop when idx_end hits the start index.
            while idx_end > idx_start and end_char <= offset[idx_end][0]: 
                idx_end -= 1

            # add token end index to list
            end_token_indices.append(idx_end)
        
    # 5. add start and end token indices to model inputs object
    model_inputs["start_positions"] = start_token_indices
    model_inputs["end_positions"] = end_token_indices

    # The defensive fix for mapping the original example index (0-999)
    # Add the expanded ID list to the output
    model_inputs["example_id"] = sample_ids
    model_inputs["answers"] = expanded_answers

    return model_inputs     


def get_train_val_test_set(
        datasets: DatasetDict,) -> tuple[
            Dataset, 
            Dataset, 
            Dataset]:
    """Splits datasets into training set, validation set and test set
    
    Args:
        datasets (DatasetDict): tokenized datasets to be split

    Return: triple of Dataset objects: training set, 
    validation set and test set
    """  
    
    # split original training set into training and internal validation
    # take 10% validation set
    train_val_set = datasets["train"].train_test_split(test_size=0.1, seed=10)

    train_set = train_val_set["train"]
    val_set = train_val_set["test"]

    # grab test set from original datasets validation key
    test_set = datasets["validation"]

    return train_set, val_set, test_set


    

