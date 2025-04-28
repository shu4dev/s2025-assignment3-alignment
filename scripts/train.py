import torch, argparse, wandb
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from cs336_alignment.instruction_fine_tuning.dataset import IFT
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

parser = argparse.ArgumentParser(description="Instruction Fine-Tuning")
parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
parser.add_argument("--weight_decay", type=float, help="weight decay", default=2e-5)
parser.add_argument("--seq_len", type=int, help="context length", default=512)
parser.add_argument("--batch_size", type=int, help="batch size", default=1)
parser.add_argument("--num_epochs", type=int, help="number of training epochs", default=1)
parser.add_argument("--device", type=str, help="gpu device", default="cuda")
parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation steps", default=16)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.bfloat16,
    #attn_implementation="flash_attention_2",
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

num_batches_epoch = len(train_dataset) // args.batch_size
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

for epoch in range(args.num_epochs):
    wandb.log({"epoch": epoch})
    for idx, batch in enumerate(tqdm(train)):
        model.train()
        losses = []
        input_ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        logits = model(input_ids).logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        losses.append(loss.item())
        if (idx + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        avg_loss = sum(losses) / len(losses)
        wandb.log({"train_loss": avg_loss})
        
    with torch.no_grad():    
        for val_idx, val_batch in enumerate(tqdm(test)):
            model.eval()
            val_losses = []
            val_input_ids = val_batch["input_ids"].to(args.device)
            val_labels = val_batch["labels"].to(args.device)
            val_logits = model(val_input_ids).logits
            val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            val_losses.append(loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            wandb.log({"val_loss": avg_val_loss})

# Save the model weights
output_dir = "Qwen/Qwen2.5-0.5B-Instruct"
model.save_pretrained(save_directory=output_dir)
tokenizer.save_pretrained(save_directory=output_dir)