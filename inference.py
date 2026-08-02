import os
import torch
import torch.nn as nn
from model.llama import Llama

def generate_text_greedy(prompt,max_new_tokens,tokenizer,model):
    model.eval()
    ip=torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits=model(ip)
        id_=torch.argmax(logits[:,-1,:],dim=-1)
        ip=torch.cat((ip,id_.unsqueeze(1)),dim=1)
    return tokenizer.decode(ip.squeeze(0).tolist())

def generate_text(prompt,max_new_tokens,tokenizer,model,temp):
    model.eval()
    ip=torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits=model(ip)
        logits=logits[:, -1, :]
        logits/=temp
        logits=torch.softmax(logits,dim=-1)
        id_=torch.multinomial(logits,num_samples=1)
        ip=torch.cat((ip,id_),dim=1)
    return tokenizer.decode(ip.squeeze(0).tolist())


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


print(generate_text_greedy(
    prompt="ఒక",
    max_new_tokens=20,
    tokenizer=tokenizer,
    model=model
))

for _ in range(5):
    print(generate_text(
        prompt="ఒక",
        max_new_tokens=30,
        tokenizer=tokenizer,
        model=model,
        temp=0.8
    ))
    print("-" * 40)