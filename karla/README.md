# Karla - The Self-Evolving Reasoner

<div align="center">

**The Self-Evolving Reasoner (Frankenstein 2.0)**

*A 3-Level Nested Learning Architecture for Efficient Reasoning*

**Target: DeepSeek-Level Reasoning on RTX 4060 Ti (16GB VRAM)**

</div>

---

## 🧠 Architecture Overview

Karla implements a novel **Nested Learning** architecture that separates learning, memory, and reasoning into three distinct levels with different update frequencies:

```
┌─────────────────────────────────────────────────────────────┐
│                         INPUT (Tokens)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              L0: Perception (Update Freq = 0)               │
│                   Qwen 2.5-1.5B 4-bit                        │
│                      [FROZEN]                                │
│                   ~2 GB VRAM                                 │
└───────────┬─────────────────────────────────────┬───────────┘
            │                                     │
            ▼                                     ▼
┌───────────────────────┐             ┌───────────────────────┐
│  L1: Memory (Low Freq)│             │   Direct Fusion       │
│    CMS Engram         │             │                       │
│    Hash-based Lookup  │             │                       │
│    [CPU/RAM]          │             │                       │
│    Delta Rule Updates │             │                       │
└───────────┬───────────┘             └───────────┬───────────┘
            │                                     │
            └─────────────────┬───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            L2: Reasoning (Update Freq = High)               │
│                CTM Head + BitNet 1.58                        │
│                   [TRAINABLE]                                │
│                  ~50 MB VRAM                                 │
│         Iterative "Thinking" with Internal Ticks            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
                       OUTPUT
```

## 🔬 Scientific Foundation

Karla integrates research from five groundbreaking papers:

| Paper | Concept | Application |
|-------|---------|-------------|
| **CTM** (Sakana AI) | Neural synchronization, internal ticks | L2 Reasoning Head |
| **Engram** (DeepSeek) | Hash-based memory, O(1) lookup | L1 Memory System |
| **POPE** (CMU) | Oracle-guided exploration | Training methodology |
| **Nested Learning** (Google) | Multi-timescale updates | Architecture principle |
| **Grokking** (OpenAI) | Late generalization | Training dynamics |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd /home/z/my-project/karla

# Install dependencies
pip install -r requirements.txt
```

### Quick Test

```bash
# Run with mock components for quick testing
python train_karla.py --mode test
```

### Full Training

```bash
# Train with full model
python train_karla.py --mode train --epochs 50 --batch-size 4

# Quick training with lite model
python train_karla.py --mode train --quick
```

### Evaluation

```bash
python train_karla.py --mode eval --checkpoint checkpoints/best_model.pt
```

## 📊 Memory Budget

| Component | VRAM | RAM | Update Frequency |
|-----------|------|-----|------------------|
| L0 (Qwen 4-bit) | ~2 GB | - | 0 (Frozen) |
| L1 (Engram) | - | Scalable | Low (Delta Rule) |
| L2 (CTM + BitNet) | ~50 MB | - | High (Backprop) |
| **Total** | **~2.5 GB** | **Variable** | - |

## 🧪 Training Phases

### Phase 1: Next-Token Prediction Warmup
- Standard language modeling objective
- Establish baseline capabilities

### Phase 2: POPE-Guided Reasoning
- Oracle prefixes for hard problems
- Mixed guided/unguided training
- Transfer from guided to unguided

### Phase 3: Fine-Tuning
- Adaptive computation time
- Grokking detection and exploitation

## 🔑 Key Features

### 1. Nested Learning
- **L0**: Frozen perception backbone (no updates)
- **L1**: Slow memory updates via Delta Rule
- **L2**: Fast reasoning updates via backprop

### 2. POPE Training
- Hard problems receive oracle hints
- Behaviors transfer to unguided problems
- Overcomes exploration challenges

### 3. Grokking Detection
- Monitors for late generalization
- Adjusts training when grokking occurs
- Weight decay critical for inducing grokking

### 4. BitNet Efficiency
- Ternary weights: {-1, 0, +1}
- Extreme memory efficiency
- Fast inference

## 📁 Project Structure

```
karla/
├── __init__.py
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── karla.py           # Main model
│   ├── l0_perception.py   # Frozen Qwen backbone
│   ├── l1_engram.py       # Hash-based memory
│   └── l2_ctm.py          # CTM reasoning head
├── training/
│   ├── __init__.py
│   └── trainer.py         # Training pipeline
├── data/
│   ├── __init__.py
│   └── dataset.py         # POPE dataset
└── utils/
    ├── __init__.py
    └── config.py          # Configuration
```

## 🎯 Training Goals

1. **Grokking**: Detect and leverage late generalization
2. **Transfer**: Oracle-guided → unguided reasoning
3. **Efficiency**: Run on consumer hardware (RTX 4060 Ti)
4. **Quality**: Achieve "DeepSeek-level" reasoning

## 📈 Monitoring

Training logs include:
- Loss curves (train/val)
- Learning rate schedule
- Internal ticks (adaptive compute)
- Certainty scores
- Grokking detection alerts

## 🔧 Configuration

Key hyperparameters:

```python
# L0: Perception
l0_model_name = "Qwen/Qwen2.5-1.5B"
l0_bits = 4  # 4-bit quantization

# L1: Memory
l1_embedding_dim = 512
l1_num_heads = 8

# L2: Reasoning
l2_hidden_dim = 512
l2_num_neurons = 256
l2_num_internal_ticks = 10
l2_use_bitnet = True

# Training
l2_lr = 3e-4
weight_decay = 0.01  # Critical for grokking!
```

## 📚 References

1. **CTM**: Darlow et al., "Continuous Thought Machines" (NeurIPS 2025)
2. **Engram**: Cheng et al., "Conditional Memory via Scalable Lookup" (DeepSeek)
3. **POPE**: Qu et al., "Learning to Reason on Hard Problems" (CMU)
4. **Nested Learning**: Behrouz et al., "Nested Learning" (Google, NeurIPS 2025)
5. **Grokking**: Power et al., "Grokking: Generalization Beyond Overfitting" (OpenAI)

## 📜 License

This project is for research purposes.

---

<div align="center">

**"We cannot solve our problems with the same thinking we used when we created them!"**  
— Albert Einstein

</div>
