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


import torch
from BPE import BPE
from data_loader import load_dataset_,create_loader
ds=load_dataset_()
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
