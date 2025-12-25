from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

MODEL_CHECKPOINT = "JetBrains/Mellum-4b-base"

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
    
    # split full set into training and testing 80/20
    train_test_set = datasets["train"].train_test_split(test_size=0.2, seed=81)

    # take 10% from train set for validation
    train_val_set = train_test_set["train"].train_test_split(test_size=0.1, seed=75)
    
    train_set = train_val_set["train"]
    val_set = train_val_set["test"]

    # test set
    test_set = train_test_set["test"]

    return train_set, val_set, test_set

def get_tokenizer():
    """loads and returns model-specific tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_tokenize_function(tokenizer):
    """Used only for calculating sample size and setting max seq length"""
    def tokenize_function(examples):

        texts = [
            f"### Instruction:\n{inst}\n\n### Response:\n{out}{tokenizer.eos_token}" 
            for inst, out in zip(examples['instruction'], examples['output'])
        ]
        return tokenizer(texts, truncation=True)
    return tokenize_function

def get_formatting_function(tokenizer):
    def format_func(examples):
        # Check if inputs are lists (batched) or strings (single example)
        if isinstance(examples['instruction'], list):
            output_texts = []
            for i in range(len(examples['instruction'])):
                text = f"### Instruction:\n{examples['instruction'][i]}\n\n### Response:\n{examples['output'][i]}{tokenizer.eos_token}"
                output_texts.append(text)
            return output_texts
        else:
            # Handle single example case
            text = f"### Instruction:\n{examples['instruction']}\n\n### Response:\n{examples['output']}{tokenizer.eos_token}"
            return text # return string for single item
            
    return format_func