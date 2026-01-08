# PyTorch from First Principles
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/docs/stable/index.html)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

> **"What I cannot create, I do not understand."** — *Richard Feynman*

This repository is a deep-dive journey into PyTorch, moving beyond high-level APIs to understand the mathematical and computational "First Principles" of Deep Learning. Each chapter consists of hands-on problems designed to break the "black box" and prepare for implementing complex research papers from scratch.

## The Philosophy
Instead of rote memorization, this series focuses on **Why?**
- **Memory:** How are tensors actually stored in RAM?
- **Math:** How do we write Einstein Summation for high-dimensional research?
- **Calculus:** How does the Autograd engine track the chain rule?
- **Architecture:** How do we turn mathematical equations into modular objects?

---

## Curriculum: Five Chapters of Depth

### [Chapter 01: Memory & Strides](./notebooks/01_Tensor_Internal_Storage.ipynb)
**The Objective:** Understand that a Tensor is just a "view" of a flat 1D array.
- **Key Concepts:** Storage, Stride, Storage Offset, and Contiguity.
- **The Research Why:** Learn why `transpose()` is zero-latency and why `view()` fails on non-contiguous memory.
- **Problem:** Manually calculate the memory index of a transposed matrix using stride math.

### [Chapter 02: Dimension Mastery (Einsum & Unfold)](./notebooks/02_dimension_mastery.ipynb)
**The Objective:** Master high-dimensional manipulation without "Dimension Hell."
- **Key Concepts:** `torch.einsum`, Broadcasting (Expansion Trick), and `unfold`.
- **The Research Why:** Implement the **Multi-Head Attention** score and **Sliding Window Convolutions** using declarative index notation.
- **Problem:** Use `unfold` to implement a manual 1D convolution without `nn.Conv1d`.

### [Chapter 03: The Autograd Engine (Calculus from Scratch)](./03_Autograd_and_Backprop.ipynb)
**The Objective:** Demystify the computational graph and the backpropagation tape.
- **Key Concepts:** `requires_grad`, `backward()`, `detach()`, and Gradient Accumulation.
- **The Research Why:** Audit PyTorch's math by comparing Autograd results against manual Chain Rule derivatives.
- **Problem:** Build a manual training loop for a single neuron without using `nn.Linear` or built-in optimizers.

### [Chapter 04: Modular Foundations (Building nn.Module)](./04_Modular_Architectures.ipynb)
**The Objective:** Transition from raw tensors to "Batch-Ready" objects.
- **Key Concepts:** `nn.Module`, `nn.Parameter`, and `state_dict`.
- **The Research Why:** Understand how PyTorch "finds" parameters inside nested modules (like Transformers).
- **Problem:** Implement a custom `MyLinear` layer using `einsum` that supports vectorized batches of data.

### [Chapter 05: The Industrial Pipeline (Datasets & Training)](./05_Data_and_Training_Loops.ipynb)
**The Objective:** Build a scalable "Conveyor Belt" for real-world research data.
- **Key Concepts:** `Dataset`, `DataLoader`, `nn.Sequential`, and `BCEWithLogitsLoss`.
- **The Research Why:** Handle data bottlenecks using multiprocessing and understand numerical stability in loss functions.
- **Problem:** Build a full end-to-end system to solve a non-linear "Circular Classification" problem using a custom MLP and data loader.

---

##  Tech Stack & Tools
- **PyTorch:** The core engine.
- **Matplotlib:** For visualizing convergence and attention heatmaps.
- 
## How to use this Repo
1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/pytorch-first-principles.git
   ```
2. **Install dependencies:**
   ```bash
   pip install -q torch matplotlib
   ```
3. **Run the Notebooks:** Start with `01_Tensor_Internal_Storage.ipynb` and follow the "Post-Mortem" analysis in each cell to understand the common pitfalls.

## Post-Mortem Analysis
Each problem in this repository is followed by a **Post-Mortem** section. This is a critical self-audit where we analyze:
1. Common errors faced (e.g., In-place updates in Autograd).
2. The underlying math or memory rule that explains the fix.
3. Where this specific concept appears in major AI papers (e.g., GANs, Transformers).



