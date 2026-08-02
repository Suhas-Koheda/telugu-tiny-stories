import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model.BPE import BPE
from model.llama import Llama
from data_loader import load_dataset_, create_loader

ds = load_dataset_()
corpus = "\n".join(sample["text"] for sample in ds)
print(len(corpus))
tokenizer=BPE(vocab_size=1000)
tokenizer.train(corpus)
dataloader=create_loader(corpus,tokenizer,max_length=128,stride=128,shuffle=False,batch_size=8)

data_iter=iter(dataloader)
ip,tgt=next(data_iter)
print("\nInputs\n",ip)
print("\ntarget\n",tgt)

print("\nInput shape:\n",ip.shape)
print("\nTarget shape:\n",tgt.shape)

token_ids=tokenizer.encode(corpus)
torch.save(token_ids,"token_ids.pt")


dataset = dataloader.dataset

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

cfg = {
    "vocab_size": 1000,      # Should match your tokenizer vocab
    "embedding_dim": 256,    # Hidden size (d_model)

    "num_layers": 6,
    "num_heads": 8,
    "num_kv_heads": 8,       # Same as num_heads for now (later change for GQA)

    "ffn_dim": 1024,         # Usually 4× embedding_dim (SwiGLU will adjust)

    "max_seq_len": 128,

    "dropout": 0.0,

    "rms_norm_eps": 1e-6,

    "rope_theta": 10000.0,

    "bias": False
}


model=Llama(cfg)
optim=torch.optim.AdamW(model.parameters(),lr=1e-3)
loss_fn=nn.CrossEntropyLoss()
model.train();

epochs=2
best_val_loss=float("inf")
start=0


if os.path.exists("latest_model.pt"):

    checkpoint = torch.load("latest_model.pt")

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optim.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start = checkpoint["epoch"] + 1

    best_val_loss = checkpoint["val_loss"]

    print(f"Resuming from epoch {start}")


device="cuda" if torch.cuda.is_available() else "cpu"
mode=model.to(device)
for epoch in range(start,epochs):
    epoch_loss_train=0
    epoch_loss_val=0
    model.train()
    for ip,tgt in tqdm(train_loader):
        ip=ip.to(device)
        tgt=tgt.to(device)
        logits=model(ip)
        logits=logits.view(-1,cfg["vocab_size"])
        tgt=tgt.view(-1)
        loss=loss_fn(logits,tgt)
        optim.zero_grad()
        loss.backward()
        optim.step()
        epoch_loss_train+=loss.item()
    model.eval()
    with torch.inference_mode():
        for ip,tgt in tqdm(val_loader):
            ip=ip.to(device)
            tgt=tgt.to(device)
            logits=model(ip)
            logits=logits.view(-1,cfg["vocab_size"])
            tgt=tgt.view(-1)
            loss=loss_fn(logits,tgt)
            epoch_loss_val+=loss.item()
    epoch_loss_train/=len(train_loader)
    epoch_loss_val/=len(val_loader)
    torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "train_loss": epoch_loss_train,
            "val_loss": epoch_loss_val,
        }, "latest_model.pt")
    if epoch_loss_val<epoch_loss_val:
        best_vl_loss=epoch_loss_val
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "train_loss": epoch_loss_train,
            "val_loss": epoch_loss_val,
        }, "best_model.pt")
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss_train:.4f} | Val Loss: {epoch_loss_val:.4f}")