# Transformer-Basic-Project

A lightweight, educational implementation of a Transformer-based next-token prediction model built from scratch in PyTorch. This project demonstrates the core ideas behind self-attention, positional encoding, causal masking, and autoregressive text generation using a small character-level vocabulary.

## Overview

This repository contains a notebook-based implementation of a minimal Transformer for next-token prediction. Instead of using a large pretrained language model, the project focuses on the foundational mechanics of a Transformer:

- token embeddings
- positional embeddings
- scaled dot-product self-attention
- causal attention masking
- residual connections
- feed-forward layers
- next-token generation in an autoregressive setting

The goal is not to produce a production-grade large language model, but to make the architecture understandable and teachable by training it on a very small corpus.

## Project Objective

The model learns to predict the next character in a sequence using a small toy corpus. Given a prompt such as:

```
system training
```

it learns to continue the sequence in a coherent way based on patterns in the training data.

## Approach and Technique

### 1. Tokenization
The project uses a character-level vocabulary rather than word-level or subword tokenization. The corpus is converted into integer token IDs using a dictionary mapping each unique character to an index.

This makes the problem simple and effective for a learning demo while still illustrating the transformer mechanism clearly.

### 2. Input Representation
Each input sequence is represented as:

- token IDs for each character in the sequence
- positional embeddings to encode token order
- learned character embeddings to represent the semantic meaning of each token

The final input to the transformer block is:

```
input = token_embedding + positional_embedding
```

### 3. Self-Attention
The self-attention mechanism computes:

- Query (Q)
- Key (K)
- Value (V)

The attention scores are computed using scaled dot-product attention:

```
attention_scores = Q @ K^T / sqrt(d_k)
```

A lower triangular causal mask is applied so that each token can only attend to itself and earlier tokens. This is essential for autoregressive generation.

### 4. Transformer Block
Each transformer block follows a typical structure:

- Layer normalization
- self-attention
- residual connection
- feed-forward network
- residual connection

This is a simplified but faithful Transformer-style block that captures the essence of the architecture.

### 5. Training Objective
The model is trained as a next-token prediction task using cross-entropy loss.

For each sequence:

- input sequence = first n tokens
- target sequence = shifted sequence (next token prediction)

The model predicts the next token for each position in the sequence, and the loss is computed over all positions.

### 6. Generation Strategy
During inference, the model generates text autoregressively:

1. provide the initial prompt
2. encode the last sequence window
3. predict the next token
4. append it to the prompt
5. repeat until the desired length is reached

This is a standard generation pattern used in language models.

## Model Architecture Summary

The notebook implements a minimal Transformer with:

- embedding dimension: 64
- sequence length: 16
- 2 transformer blocks
- causal self-attention
- feed-forward hidden dimension: 4 × embedding dimension
- final linear output head to vocabulary size

## Dataset and Corpus

The toy corpus used in the notebook is a small set of sentences:

```
system training is simple.
transformers are powerful.
self attention helps us predict the next token easily.
learning deep learning is fun!
```

This small corpus is intentionally compact so that training is fast and the learning process is easy to visualize.

## Metrics Used

Because this project is a next-token classification problem, the most relevant indicators are not regression metrics such as R² score or Pearson correlation. These are useful for continuous prediction tasks, but this model predicts a categorical class (the next character) from a vocabulary, so the primary metrics are:

### Core Training Metrics
- Loss
- Perplexity
- Token accuracy
- Cumulative tokens processed

### Why not R² / correlation?
R², Pearson correlation, and similar regression metrics are designed for continuous target variables. In this project, the target is a token index from a finite vocabulary. The natural objective is token-level classification, so cross-entropy loss and token accuracy are more appropriate metrics.

## Observed Training Results

The notebook logs the following final performance:

- Initial Loss: 3.4786
- Final Loss: 0.1232
- Loss Reduction: 96.46%
- Final Perplexity: 1.131
- Final Token Accuracy: 0.9736
- Best Loss Epoch: 150
- Total Tokens Processed: 295,200

These values show that the model learns the toy corpus effectively and converges rapidly.

## Example Output

A sample prompt and generated output from the notebook:

```
Prompt: 'system training'
Completion: 'system training is simple.
transformers are powerful.
self attent'
```

This demonstrates the model is able to produce text that follows the style and structure of the training corpus, even though it is trained on a very small dataset.

## File Structure

- `Transformer_project.ipynb` — main notebook containing the full model implementation and training process
- `Validation Graph.png` — visualization of the training convergence and loss behavior
- `README.md` — project documentation and summary

## Key Observations

- Training loss drops sharply in early epochs.
- Perplexity decreases significantly as the model learns.
- Token accuracy rises to nearly 97% by the end of training.
- The model learns the basic grammar and sentence patterns in the toy corpus.
- The project serves as an interpretable demonstration of how attention-based sequence models work.

## Use Cases and Learning Value

This repository is useful for:

- learning the fundamentals of Transformers
- understanding self-attention and positional embeddings
- experimenting with causal generation
- visualizing convergence and token-level learning behavior

## Requirements

This project uses:

- Python
- PyTorch
- NumPy
- Matplotlib

## How to Run

Open the notebook in Jupyter or Google Colab and execute all cells in order.

Example workflow:

```bash
jupyter notebook Transformer_project.ipynb
```

## Conclusion

This project is a compact and approachable implementation of a Transformer for next-token prediction. It demonstrates the essential design patterns behind modern language models while keeping the scale small enough for educational study and experimentation.

It is particularly effective for understanding how a model learns sequence structure through attention, position encoding, and autoregressive generation.

---

This repository is intended as a beginner-friendly Transformer example and is best suited for educational and experimentation purposes rather than production-scale NLP workloads.
