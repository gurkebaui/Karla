#  Karla C1: The Adaptive Brain for Physical AI

![NVIDIA Cosmos](https://img.shields.io/badge/Powered_by-NVIDIA_Cosmos_Reason2-76B900?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

**Karla C1** is a novel, continuous-learning architecture designed for Physical AI and embodied agents. Built on top of **NVIDIA Cosmos-Reason2-2B**, it solves the problem of catastrophic forgetting in robotics by implementing a **Nested Learning Architecture** with *Surprise-Based Plasticity*.

##  The Vision: Physical AI on the Edge
Robots in the real world encounter new tools and constraints daily. Karla allows an agent to learn a new physical tool or API *on the fly* during inference, modifying its own weights in seconds on consumer hardware (e.g., RTX 4060 Ti), without ever forgetting its foundational knowledge.

## Architecture: Nested Learning
Instead of a standard transformer pipeline, Karla acts as a multi-frequency brain:
1. **L0 (The Subconscious): NVIDIA Cosmos-Reason2-2B (Frozen).** Provides world-class reasoning, semantic understanding, and syntax. Loaded in 4-bit to save VRAM.
2. **L1 (The RAM): Dynamic Knowledge MoE (64 Experts).** A fast, GPU-based Mixture of Experts. It updates *live during inference* using Surprise-Based Plasticity (Delta Rule), memorizing new facts and tool syntaxes instantly.
3. **L2 (The Frontal Lobe): Continuous Thought Machine (CTM).** A parallel sequence-level reasoning module. It takes Cosmos's hidden states + L1 knowledge and "thinks" for $T$ internal ticks before outputting an action plan. This part needs more Work, as it as of now can lead to the model repeating tokens. 

## Quickstart
```bash
pip install torch transformers datasets pandas accelerate bitsandbytes
cd Karla
python chat.py
```

## TODO
-Train Longer
-Make L1 Inference Learning Intensity variable
-Give the Model controll over itself
-Fix stuttering


## Datasets used:
@misc{Hermes_Reasoning_Tool_Use,
  title  = {Hermes Tool Use Reasoning},
  author = {interstellarninja},
  year   = {2025},
  howpublished = {\url{https://huggingface.co/datasets/interstellarninja/hermes_reasoning_tool_use}},
  note   = {Apache-2.0}
}

crownelius/Opus-4.6-Reasoning-2100x-formatted

ronantakizawa/github-top-code
