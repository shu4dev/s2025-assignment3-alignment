import torch, argparse, wandb, gc
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from cs336_alignment.instruction_fine_tuning.dataset import IFT
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR

parser = argparse.ArgumentParser(description="Instruction Fine-Tuning")
parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
parser.add_argument("--weight_decay", type=float, help="weight decay", default=2e-5)
parser.add_argument("--seq_len", type=int, help="context length", default=512)
parser.add_argument("--batch_size", type=int, help="batch size", default=8)
parser.add_argument("--num_epochs", type=int, help="number of training epochs", default=100)
parser.add_argument("--device", type=str, help="gpu device", default="cuda")
parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation steps", default=4)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

model.to('cuda')


optimizer = optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)

train_dataset = IFT(tokenizer, "data/tuning/train.jsonl", args.seq_len, True)
test_dataset = IFT(tokenizer, "data/tuning/test.jsonl", args.seq_len, False)

train = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=8,
    pin_memory=True,
)

test = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    num_workers=8,
    pin_memory=True,
)

scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.001, last_epoch=-1)

wandb.init(
    project="Qwen2.5-0.5B Instruction-Fine-Tuning",
    config={
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_epochs": args.num_epochs,
    },
)

def val():
    model.eval()
    losses = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test)):
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            logits = model(input_ids).logits
            loss = F.cross_entropy(logits, labels)
            losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    wandb.log({"val_loss": loss.item()})
    return avg_loss

for epoch in range(args.num_epochs):
    for idx, batch in enumerate(tqdm(train)):
        model.train()
        input_ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        logits = model(input_ids).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        
        if (idx + 1) % args.gradient_accumulation_steps == 0:
            validation_loss = val()
            wandb.log({"train_loss": loss.item()})
            wandb.log({"validation_loss": validation_loss})
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()



# Save the model weights
output_dir = "Qwen/Qwen2.5-0.5B-Instruct"
model.save_pretrained(save_directory=output_dir)
tokenizer.save_pretrained(save_directory=output_dir)