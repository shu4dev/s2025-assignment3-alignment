import os
import json
import gzip
import torch
import torch.nn.functional as F
import random
from transformers import PreTrainedTokenizer, PreTrainedModel
from huggingface_hub import snapshot_download

def dpo_dataset():
    base_path = "data/hh-rlhf"
    full_path = lambda x: os.path.join(base_path, x) 
    
    paths = {
        "harmless-base": full_path("harmless-base/train"),
        "helpful-base": full_path("helpful-base/train"),
        "helpful-online": full_path("helpful-online/train"),
        "helpful-rejection-sampled": full_path("helpful-rejection-sampled/train")
    }
    def multi_turn(sample):
        if len(sample["chosen"].split("\n\nHuman: ")) > 2:
            return None
        prompt_chosen = sample["chosen"].split("\n\nHuman: ")[1].split("\n\nAssistant: ")
        prompt = prompt_chosen[0]
        chosen = prompt_chosen[1]
        prompt_rejected = sample["rejected"].split("\n\nHuman: ")[1].split("\n\nAssistant: ")
        assert prompt == prompt_rejected[0]
        rejected = prompt_rejected[1]
        return (prompt, chosen, rejected)
    
    def load_jsonl(path):
        result = []
        with gzip.open(f"{path}.jsonl.gz", 'rt', encoding='utf-8') as file:
            for line in file:
                result.append(json.loads(line))
        return result
    
    results = []
    for dataset_name, path in paths.items():
        print(f"Processing {dataset_name} dataset...")
        dataset = load_jsonl(path)
        for sample in dataset:
            clean_sample = multi_turn(sample)
            if clean_sample is None:
                continue
            prompt, chosen, rejected = clean_sample
            result = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "dataset": dataset_name
            }
            results.append(result)

    out_paths = "data/anthropic/train/train"
    with open(f"{out_paths}.jsonl", 'w', encoding='utf-8') as f:
        for result in results:
            json_line = json.dumps(result)
            f.write(json_line + '\n')

prompt_format = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
{response}"""

def get_logprob(
    model, 
    prompt, 
    response, 
    tokenizer):
    input = torch.tensor(tokenizer.encode(prompt_format.format(prompt=prompt, response=response)) + [tokenizer.eos_token_id]).unsqueeze(0).to(model.device)
    labels = input.clone()[:, 1:]
    logits = model(input).logits[:, :-1, :]
    logits = F.log_softmax(logits, dim=-1)
    return torch.gather(logits, 2, labels.unsqueeze(-1)).squeeze(-1).sum(-1)

def dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:

    with torch.no_grad():
        pi_ref_chosen = get_logprob(lm_ref, prompt, response_chosen, tokenizer)
        pi_ref_rejected = get_logprob(lm_ref, prompt, response_rejected, tokenizer)
    pi_chosen = get_logprob(lm, prompt, response_chosen, tokenizer)
    pi_rejected = get_logprob(lm, prompt, response_rejected, tokenizer)
    return -F.logsigmoid(beta * (pi_chosen - pi_rejected + pi_ref_rejected.to(lm.device) - pi_ref_chosen.to(lm.device)))