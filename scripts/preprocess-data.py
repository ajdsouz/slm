from datasets import load_dataset, DatasetDict, concatenate_datasets
import argparse
from functools import partial
from transformers import AutoTokenizer, PreTrainedTokenizerBase

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-name", type=str)
parser.add_argument("--subset", type=str)
# parser.add_argument("--split", type=float)
parser.add_argument("--num-proc", type=int)
parser.add_argument("--input-column", type=str)
parser.add_argument("--test-ratio", type=float)
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--data-dir", type=str)

args = parser.parse_args()


def preprocess_dataset(
        dataset: DatasetDict, 
        tokenizer: PreTrainedTokenizerBase, 
        input_column: str,
        num_proc: int
    ) -> DatasetDict:
    

    def process(batch):

        tokenized = tokenizer(
            batch[input_column],
            padding=False,
            return_tensors=None
        )
        
        return {
            'input_ids': tokenized['input_ids']
        }
    
    
    dataset = dataset.map(
        process,
        batched=True,
        num_proc=num_proc,
        remove_columns=input_column
    )

    return dataset


def create_datasets(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = load_dataset(args.dataset_name, args.subset)
    print(f"using {args.tokenizer} on {args.dataset_name}")

    processed_dataset = preprocess_dataset(
        dataset = dataset,
        tokenizer=tokenizer,
        input_column=args.input_column,
        num_proc=args.num_proc
    )

    # processed_dataset = processed_dataset['train'].train_test_split(test_size=args.test_ratio)
    processed_dataset.save_to_disk(args.data_dir)


create_datasets(args=args)