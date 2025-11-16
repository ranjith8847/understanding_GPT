#  Mini GPT Language Model (PyTorch)

A clean, educational, and fully-from-scratch implementation of a **GPT-style autoregressive language model** trained on *The Wizard of Oz* text.  
This project follows the architecture explained in Karpathy’s "nanoGPT" videos but expands it with clearer structure, modular components, and training utilities.

---

## 🚀 Features

- **Full Transformer Decoder** implementation:
  - Multi-Head Self-Attention  
  - Causal Masking  
  - Feed-Forward Network  
  - LayerNorm + Residual Connections  
- **Character-Level Tokenizer** (no external dependencies)
- **Text Generation** from a trained checkpoint
- **Training Loop** with:
  - Mini-batch sampling  
  - Evaluation on validation set  
  - AdamW optimizer  
- Trains on the **Wizard of Oz** dataset (provided as plain text)

---

## 📂 Project Structure

```
.
├── wizard_of_oz.txt
├── gpt_model.py # (your main file)
└── README.md
```


---

## ⚙️ Hyperparameters

| Parameter          | Value         |
|-------------------|---------------|
| block_size        | 64            |
| batch_size        | 128           |
| n_embed           | 384           |
| n_head            | 4             |
| n_layer           | 4             |
| dropout           | 0.2           |
| max_iters         | 3000          |
| eval_iters        | 100           |
| learning_rate     | 3e-4          |

---

## 🏗️ Model Architecture

### 1. **Embedding Layer**
- Token embeddings  
- Positional embeddings  

### 2. **Stack of Decoder Blocks**
Each block contains:
- Multi-Head Self-Attention  
- Feed-Forward network  
- Residual connections  
- LayerNorm  

### 3. **Final Classification Head**
- Linear layer → Vocabulary logits  

### 4. **Generation**
Greedy sampling using `torch.multinomial`.

---

## 📊 Training

Runs for **3000 iterations**, logging train/val loss:

```
step: 0, train: 4.43, val: 4.42
step: 500, train: 1.51, val: 1.68
step: 1500, train: 1.12, val: 1.49
step: 2900, train: 0.82, val: 1.59
```
---

## 🔍 How It Works

### **Batching**
Random slices of the text are sampled to create input (X) and target (Y) training sequences of length `block_size`.

### **Loss Calculation**
Cross-entropy over all tokens in the batch.

### **Causal Mask**
Ensures the model cannot “peek” into the future by masking upper-triangular attention scores.

### **Generation**
Given a prompt, repeatedly:
- Compute logits  
- Apply softmax  
- Sample next token  
- Append to sequence  

---

## 🖥️ Training the Model

```bash
python gpt_model.py

```

# 📝 Example: Generate Text

Add at the bottom of your script:
```
start = torch.zeros((1,1), dtype=torch.long).to(device)
out = model.generate(start, max_new_tokens=200)
print(decode(out[0].tolist()))
```
