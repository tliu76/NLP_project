# Machine Translation with Seq2Seq and Transformers

A PyTorch implementation of machine translation models for English–German translation, built as part of CS 7643 Deep Learning (OMSCS, Georgia Tech).

## Overview

This project implements and compares three neural machine translation architectures on the [Multi30K](https://github.com/multi30k/dataset) dataset:

1. **Naive LSTM** — custom LSTM cell built from scratch with `nn.Parameter`
2. **Seq2Seq with Attention** — RNN/LSTM encoder-decoder with cosine similarity attention
3. **Transformer (Encoder-only)** — single-layer Transformer encoder with multi-head self-attention
4. **Full Transformer** — complete encoder-decoder Transformer using `nn.Transformer`

## Dataset

**Multi30K** — 31,014 English–German sentence pairs from image descriptions:
- Train: 29,000 pairs
- Validation: 1,014 pairs
- Test: 1,000 pairs

Example:
- EN: `Two young, White males are outside near many bushes.`
- DE: `Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.`

## Project Structure

```
NLP_project/
├── Machine_Translation.ipynb      # Main notebook (training, evaluation, translation)
└── models/
    ├── naive/
    │   └── LSTM.py                # Custom LSTM from scratch
    ├── seq2seq/
    │   ├── Encoder.py             # RNN/LSTM encoder
    │   ├── Decoder.py             # RNN/LSTM decoder with optional attention
    │   └── Seq2Seq.py             # Seq2Seq wrapper
    └── Transformer.py             # TransformerTranslator & FullTransformerTranslator
```

## Models

### 1. Naive LSTM (`models/naive/LSTM.py`)
Implements the four LSTM gates using raw `nn.Parameter` tensors:

$$i_t = \sigma(x_t W_{ii} + b_{ii} + h_{t-1} W_{hi} + b_{hi})$$
$$f_t = \sigma(x_t W_{if} + b_{if} + h_{t-1} W_{hf} + b_{hf})$$
$$g_t = \tanh(x_t W_{ig} + b_{ig} + h_{t-1} W_{hg} + b_{hg})$$
$$o_t = \sigma(x_t W_{io} + b_{io} + h_{t-1} W_{ho} + b_{ho})$$
$$c_t = f_t \odot c_{t-1} + i_t \odot g_t, \quad h_t = o_t \odot \tanh(c_t)$$

### 2. Seq2Seq with Attention (`models/seq2seq/`)
- **Encoder**: Embedding → RNN/LSTM → Linear–ReLU–Linear projection to decoder hidden size
- **Decoder**: Embedding → cosine similarity attention → RNN/LSTM → Linear → LogSoftmax
- **Attention**: $\text{cosine}(q, K) = \frac{q \cdot K^\top}{|q||K|}$, used to compute a context vector as a weighted sum of encoder outputs

### 3. Transformer Encoder (`TransformerTranslator`)
Single-layer Transformer encoder for translation:
- Word + positional embeddings
- 2-head self-attention (scaled dot-product): $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$
- Add & LayerNorm after attention and feed-forward sublayers
- Feed-forward: Linear(H → 2048) → ReLU → Linear(2048 → H)
- Final linear projection to vocabulary size

### 4. Full Transformer (`FullTransformerTranslator`)
Encoder-decoder Transformer built on `nn.Transformer`:
- Separate source/target embeddings with positional encodings
- Causal (autoregressive) target mask during training
- Autoregressive decoding at inference via `generate_translation`

## Training

All models are trained on Google Colab (GPU). Common setup:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau |
| Loss | CrossEntropyLoss (pad ignored) |
| Batch size | 128 |
| Max sequence length | 20 |

**Seq2Seq hyperparameters:**

| Hyperparameter | Value |
|---|---|
| Embedding size | 256 |
| Hidden size | 256 |
| Dropout | 0.3 |
| Model type | LSTM |
| Epochs | 20 |
| Learning rate | 5e-4 |

**Full Transformer hyperparameters:**

| Hyperparameter | Value |
|---|---|
| Hidden dim | 128 |
| Attention heads | 2 |
| Feed-forward dim | 2048 |
| Encoder/Decoder layers | 2 |
| Dropout | 0.2 |
| Epochs | 25 |
| Learning rate | 5e-4 |

## Requirements

```
torch
numpy
tqdm
```

Run on Python 3.12. Designed for Google Colab (GPU recommended).

## Usage

Open `Machine_Translation.ipynb` in Google Colab, mount your Drive, and run cells sequentially. The notebook covers:
1. Data download and preprocessing (Multi30K)
2. LSTM unit test
3. Seq2Seq training and evaluation
4. Transformer training and evaluation
5. Translation output comparison

## Results

Sample translations from the Full Transformer on test set:

| Reference | Predicted |
|---|---|
| a man in an orange hat starring at something | a man in a hat hat is \<unk\> |
| a girl in karate uniform breaking a stick with a front kick | a girl in a pink shirt is a a with a |
| a group of people standing in front of an igloo | a group of people standing in front of a |

## References

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) — Vaswani et al., 2017
- [Sequence to Sequence Learning](https://arxiv.org/abs/1703.03906) — Sutskever et al., 2014
- [Multi30K Dataset](https://arxiv.org/abs/1605.00459) — Elliott et al., 2016
- [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — Olah, 2015
