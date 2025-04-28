import torch, argparse, wandb
from tqdm import tqdm
from cs336_alignment.dpo import dpo_loss
import json
import random
import torch.nn.functional as F
import random
from transformers import PreTrainedTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

parser = argparse.ArgumentParser(description="Instruction Fine-Tuning")
parser.add_argument("--lr", type=float, help="learning rate", default=1e-6)
parser.add_argument("--weight_decay", type=float, help="weight decay", default=2e-5)
parser.add_argument("--seq_len", type=int, help="context length", default=512)
parser.add_argument("--batch_size", type=int, help="batch size", default=64)
parser.add_argument("--num_epochs", type=int, help="number of training epochs", default=10)
parser.add_argument("--device", type=str, help="gpu device", default="cuda")
parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation steps", default=4)
parser.add_argument("--beta", type=float, help="beta for DPO loss", default=0.1)
args = parser.parse_args()

if (torch.cuda.device_count() >= 2):
    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.float32,
    #attn_implementation="flash_attention_2",
)
model.to(model_device)
model.train()

ref_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.float32,
    #attn_implementation="flash_attention_2",
)
ref_model.to(ref_device)
ref_model.eval()

optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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

def load_jsonl(file_path):
    data = []
    with open(f"{file_path}.jsonl", 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

train_set = load_jsonl("data/anthropic/train/train") 
valid_set = load_jsonl("data/anthropic/test/test")
train_set = train_set[:800]
valid_set = valid_set[:200]
random.seed(1)
random.shuffle(train_set)

num_batches_epoch = len(train_set) // args.batch_size
num_batches = num_batches_epoch * args.num_epochs
warmup_steps = int(0.03 * num_batches)

warmup_scheduler = LinearLR(
    optimizer,
    start_factor=1e-4,
    end_factor=1.0,
    total_iters=warmup_steps
)

cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=(num_batches - warmup_steps)
)

lr_scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps]
)

wandb.init(
    project="Qwen2.5-0.5B DPO",
    config={
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_epochs": args.num_epochs,
    },
)

for epoch in range(args.num_epochs):
    losses = []
    val_losses = []
    for idx, sample in enumerate(tqdm(train_set)):
        model.train()
        optimizer.zero_grad()
        prompt, chosen, rejected = sample["prompt"], sample["chosen"], sample["rejected"]
        loss = dpo_loss(model, ref_model, tokenizer, args.beta, prompt, chosen, rejected)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    avg_loss = sum(losses) / len(losses)
    wandb.log({"train_loss": avg_loss})
    for val_idx, val_sample in enumerate(tqdm(valid_set)):
        model.eval()
        optimizer.zero_grad()
        prompt, chosen, rejected = val_sample["prompt"], val_sample["chosen"], val_sample["rejected"]
        with torch.no_grad():
            val_loss = dpo_loss(model, ref_model, tokenizer, args.beta, prompt, chosen, rejected)
            val_losses.append(val_loss.item())
    avg_val_loss = sum(val_losses) / len(val_losses)
    wandb.log({"val_loss": avg_val_loss})


# Save the model weights
output_dir = "Qwen/Qwen2.5-0.5B-DPO"
model.save_pretrained(save_directory=output_dir)
tokenizer.save_pretrained(save_directory=output_dir)