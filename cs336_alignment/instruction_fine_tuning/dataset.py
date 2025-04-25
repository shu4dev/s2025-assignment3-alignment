import torch
import json
import random
from torch.utils.data import Dataset

class IFT(Dataset):
    def __init__(self, tokenizer, dataset_pth, seq_length, shuffle):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.tokens = []
        self.labels = []

        with open(dataset_pth, 'r', encoding='utf-8') as f:                
            for line in f:
                data = json.loads(line)
                instruction = data["prompt"]
                response = data["response"]
                prompt = (
                    f"Below is an instruction that describes a task. Write a response that appropriately completes the request."
                    f"\n\n### Instruction:\n{instruction}"
                    f"\n\n### Response:\n{response}"
                )
                self.tokens.append(prompt)

        self.tokens = '<|end_of_text|><|begin_of_text|>'.join(self.tokens)
        self.tokens = tokenizer.encode(self.tokens, truncation=True)
        num_samples = (len(self.tokens)-1)//seq_length
        self.labels = [self.tokens[seq_length*i+1:seq_length*(i+1)+1] for i in range(num_samples)]
        self.tokens = [self.tokens[seq_length*i:seq_length*(i+1)] for i in range(num_samples)]

        if shuffle:
            random.shuffle(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError
        return {
            "input_ids": torch.tensor(self.tokens[i], dtype=torch.long),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }

# if __name__ == "__main__":
#     sft_sample_path = "tests/fixtures/sft_sample.jsonl"
#     expected_examples_path = "tests/fixtures/tokenized_sft_sample.json"
#     with open(expected_examples_path) as f:
#         expected_examples = json.load(f)
#         print(len(expected_examples))
#
#     tokenizer = AutoTokenizer.from_pretrained("tests/fixtures/Meta-Llama-3-8B")
#     seq_length = 32
#     ift = IFT(tokenizer, sft_sample_path, seq_length, shuffle=False)
#     print(len(ift))