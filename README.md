# Telugu Tiny Stories Model

A lightweight LLaMA-architecture language model implemented from scratch in PyTorch, trained on the Telugu subset of the TinyStories dataset (`neuralnets/multilingual-tinystories`).

## Project Overview

This repository implements a custom byte-pair encoding (BPE) tokenizer and a LLaMA-based Transformer language model for generating stories in Telugu.

### Key Architecture Components

- **Custom BPE Tokenizer (`model/BPE.py`)**: Subword tokenization tailored for Telugu text.
- **RMSNorm (`model/RMSNorm.py`)**: Root Mean Square Layer Normalization.
- **RoPE (`model/ROPE.py`)**: Rotary Position Embeddings for relative position encoding.
- **Multi-Head Attention (`model/MHA.py`)**: Causal self-attention with RoPE positional embeddings.
- **SwiGLU Activation (`model/SwiGLU.py`)**: Swish-Gated Linear Unit feed-forward network.
- **Transformer Block (`model/TransformerBlock.py`)**: Residual pre-norm transformer layer combining MHA and SwiGLU.
- **LLaMA Model (`model/llama.py`)**: Complete decoder-only Transformer language model.

## Repository Structure

```
├── data_loader.py       # Dataset processing & DataLoader creation
├── inference.py         # Text generation routines (greedy & sampling)
├── train.py             # Training loop with validation and checkpointing
├── model/               # Core neural network modules
│   ├── __init__.py
│   ├── BPE.py           # Custom Byte-Pair Encoder implementation
│   ├── Embedding.py     # Embedding layer initialization
│   ├── MHA.py           # Multi-Head Attention module
│   ├── RMSNorm.py       # RMS Normalization module
│   ├── ROPE.py          # Rotary Positional Embedding module
│   ├── SwiGLU.py        # SwiGLU activation module
│   ├── TransformerBlock.py # Transformer decoder layer
│   └── llama.py         # Full LLaMA architecture model
```

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- `datasets` (HuggingFace)
- `tqdm`

Install dependencies:
```bash
pip install torch datasets tqdm
```

### Training

To train the BPE tokenizer, preprocess dataset, and start model training:
```bash
python train.py
```

### Text Generation

To generate text using greedy search or temperature sampling:
```python
from model.llama import Llama
from model.BPE import BPE
from inference import generate_text, generate_text_greedy
```
