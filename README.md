

# Karla C1: The Adaptive Brain for Physical AI

**Karla C1** is a novel, continuous-learning architecture designed for Physical AI and embodied agents. Built on top of **NVIDIA Cosmos-Reason2-2B**, it solves the problem of catastrophic forgetting in robotics by implementing a **Nested Learning Architecture** [1] with *Surprise-Based Plasticity*.

## The Vision: Physical AI on the Edge

Robots in the real world encounter new tools and constraints daily. Karla allows an agent to learn a new physical tool or API *on the fly* during inference, modifying its own weights in seconds on consumer hardware (e.g., RTX 4060 Ti), without ever forgetting its foundational knowledge.

## Architecture: Nested Learning

Instead of a standard transformer pipeline, Karla acts as a multi-frequency brain based on the Nested Learning (NL) paradigm [1]:

1. **L0 (The Subconscious): NVIDIA Cosmos-Reason2-2B (Frozen).** Provides world-class reasoning, semantic understanding, and syntax. Loaded in 4-bit to save VRAM.
2. **L1 (The RAM): Dynamic Knowledge MoE (64 Experts).** A fast, GPU-based Mixture of Experts. It updates *live during inference* using Delta Gradient Descent [1], memorizing new facts and tool syntaxes instantly.
3. **L2 (The Frontal Lobe): Continuous Thought Machine (CTM).** Based on Sakana AI's research [2], this is a parallel sequence-level reasoning module. It takes Cosmos's hidden states + L1 knowledge and "thinks" for $T$ internal ticks before outputting an action plan.

> [!WARNING]
> The L2 module currently faces stability issues where the CTM's internal dynamics can lead to token repetition during long reasoning chains.

## Quickstart

```bash
pip install torch transformers datasets pandas accelerate bitsandbytes
cd Karla
python chat.py

```

## TODO

* [ ] Train Longer
* [ ] Make L1 Inference Learning Intensity variable
* [ ] Give the model control over itself
* [ ] Fix stuttering

## Datasets Used

* `interstellarninja/hermes_reasoning_tool_use`
* `crownelius/Opus-4.6-Reasoning-2100x-formatted`
* `ronantakizawa/github-top-code`

## References

[1] Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V. (2025). **Nested Learning: The Illusion of Deep Learning Architecture**. *Google Research*. [arXiv:2512.24695](https://arxiv.org/abs/2512.24695)

[2] Darlow, L., Regan, C., Risi, S., Seely, J., & Jones, L. (2025). **Continuous Thought Machines**. *Sakana AI*. [arXiv:2505.05522](https://arxiv.org/abs/2505.05522)
