
<div align="center">
  <!-- Badges: Replace with your own. shields.io is a great tool -->
  <img src="https://img.shields.io/badge/Framework-PyTorch-purple?style=for-the-badge&logo=pytorch" alt="Framework Badge">
  <img src="https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python" alt="Language Badge">
  <img src="https://img.shields.io/badge/Paradigm-Lifelong_Learning-orange?style=for-the-badge&logo=openai" alt="Paradigm Badge">
  <img src="https://img.shields.io/github/stars/cotix-ai/MyGO?style=for-the-badge&color=gold" alt="Stars Badge">
</div>

<br>

<h1 align="center">
  MyGO: Memory Yielding Generative Offline-consolidation for Lifelong Learning Systems
</h1>

<p align="center">
  <i>Simulating the brain's "wake-sleep" cycle for continual learning without forgetting.</i>
</p>

<br>

>[!IMPORTANT]
> **Core Idea**: MyGO is a neuro-inspired lifelong learning framework that leverages generative replay (during a "sleep" phase) to consolidate old knowledge, effectively mitigating catastrophic forgetting when learning new tasks.

## Table of Contents

- [✨ Introduction](#-introduction)
- [💡 Core Design Philosophy](#-core-design-philosophy)
- [🧠 Core Architecture](#-core-architecture)
- [🧩 Core Components Explained](#-core-components-explained)
    - [Component 1: Neocortex Net](#component-1-neocortex-net)
    - [Component 2: Generative Memory Module](#component-2-generative-memory-module)
    - [Component 3: Offline Consolidation (Sleep Phase)](#component-3-offline-consolidation-sleep-phase)
- [🔄 Workflow](#-workflow)
- [🚀 Unique Advantages & Innovations](#-unique-advantages--innovations)
- [🛠️ Quick Start](#️-quick-start)
- [🤝 How to Contribute](#-how-to-contribute)
- [📄 License](#-license)

<br>

---

## ✨ Introduction

This project introduces **MyGO (Memory Yielding Generative Offline-consolidation)**, a novel framework that significantly enhances the capabilities of **neural networks in sequential task learning** by combining a **multi-task architecture** with powerful **Generative Adversarial Networks (GANs)**.

**MyGO** redefines the **continual learning process** by treating it as an **alternating cycle of "wakeful learning" and "sleep consolidation"**, rather than a simple sequential fine-tuning process. It overcomes the limitations of traditional methods (like direct fine-tuning), which suffer from catastrophically forgetting old knowledge when learning new information. This architecture synergizes the **rapid adaptation to new tasks** with the **generative replay of old knowledge**, creating a highly robust system that can continuously learn and grow without accessing old data.

<br>

---

## 💡 Core Design Philosophy

**MyGO** is not just another **continual learning algorithm**; it represents a fundamental shift in how we organize the **model training process**. We believe that achieving true lifelong learning requires systems capable of **periodically "recalling" and "consolidating" past learnings without relying on the original data**, much like the human brain.

> "True intelligence lies not only in learning quickly but also in forgetting slowly. By simulating memory consolidation during sleep, we enable AI to achieve this."

This design is engineered to overcome the inherent limitations of traditional methods in solving multi-step, sequential problems, where adapting to a new task often comes at the cost of past capabilities.

<br>

---

## 🧠 Core Architecture

The **Wake-Sleep Cycle** is the cornerstone of the **MyGO** architecture and the **"single source of truth"** for the entire **knowledge accumulation and consolidation process**. This mechanism liberates the system from the shackles of **catastrophic forgetting**.

**Core Functionality:**
The system operates by coordinating a "team" of specialized modules, each with a clear responsibility:
1.  **Wake Phase**: **Rapidly learn new knowledge**. The system focuses on the current task, freezing the general feature extractor and training only a new task-specific "head," while also training a generative model to "memorize" the current task's data distribution.
2.  **Sleep Phase**: **Consolidate all knowledge offline**. The system enters an "offline" mode, using the generative models from all tasks to produce "pseudo-data" (dreams). It then uses knowledge distillation to integrate all old and new knowledge into a unified, general feature extractor.
3.  **Memory Storage**: **Efficiently compress experience**. Instead of storing massive old datasets, the system retains only lightweight generative models (`G_mem`), dramatically reducing storage costs.

Thus, every model update is not a brute-force overwrite of old knowledge but a thoughtful integration of knowledge, blending insights from new tasks with replays of past experiences.

<br>

---

## 🧩 Core Components Explained

The different components in **MyGO** have distinct roles, working together through a clear division of labor to achieve a holistic, intelligent process.

### Component 1: Neocortex Net (Role: The Learner)
*   **Objective:** To serve as the central knowledge repository, containing a shared feature extractor (`features`) and multiple task-specific classifiers (`classifiers`).
*   **Implementation:** In the `Neocortex_Net` class, the `features` module learns cross-task, general-purpose representations. Whenever a new task is encountered, the `add_task_head` method dynamically adds a new classifier. This multi-head architecture is fundamental to isolating and organizing knowledge from different tasks.

### Component 2: Generative Memory Module (Role: The Memory Encoder)
*   **Objective:** To create a compact, generative model for each task's data distribution, enabling future "replay" of this data without needing to store the original samples.
*   **Implementation:** The `Generator` and `Discriminator` classes form a conditional Generative Adversarial Network (cGAN). During the wake phase, it is trained to mimic the data of the current task. Once trained, this lightweight `Generator` (`G_mem`) becomes a permanent memory proxy for that task.

### Component 3: Offline Consolidation (Sleep Phase) (Role: The Knowledge Integrator)
*   **Objective:** To act as the heart of the system, integrating new and old knowledge to update the shared feature extractor, making it proficient across all seen tasks and thus combating forgetting.
*   **Implementation:** The `sleep_phase` function is the core of this process. It creates a "teacher" model (a temporary model containing the latest task knowledge) and a "student" model (the main `M_ctx`). By sampling "dream" data from all past `G_mem`s and using the teacher's outputs to guide the student's learning (knowledge distillation), it achieves a forgetting-free update of the shared feature extractor.

<br>

---

## 🔄 Workflow

The operation of **MyGO** follows a clear, iterative "wake-sleep" cycle that simulates a structured learning and consolidation process:

1.  **Wake Phase - Learning:** When a new task `T` arrives, the system freezes the shared feature extractor `M_ctx.features`. It trains only the new task head `M_ctx.classifiers[T]` to rapidly adapt to the new data.
2.  **Wake Phase - Memorizing:** Concurrently, a generative model `G_mem[T]`, specifically designed for task `T`, is trained to capture its data distribution.
3.  **Sleep Phase - Dreaming:** The system enters the sleep phase. It randomly samples "pseudo" data by drawing from the generative models of the current and all past tasks (`G_mem[0]...G_mem[T]`).
4.  **Sleep Phase - Consolidation:** Using a "teacher" model that embodies all knowledge (old and new), the "student" model `M_ctx`'s shared feature extractor is trained via knowledge distillation (using MSE loss). This allows `M_ctx` to assimilate all knowledge without accessing any real old data.
5.  **Loop:** The system "wakes up," ready for the next task, and repeats the cycle. The final `M_ctx` model maintains high performance across all tasks it has learned.

<br>

---

## 🚀 Unique Advantages & Innovations

While existing methods like **Finetuning** can adapt a model to new tasks, they do so at the cost of **catastrophic forgetting**, where the model completely loses its ability to perform old tasks.

**This is precisely the gap that MyGO aims to explore and solve.**

**MyGO**, through its unique **generative replay and knowledge distillation** architecture, offers the following advantages:

*   **Significant Resistance to Catastrophic Forgetting:** By "rehearsing" old tasks during the sleep phase, the model continuously reinforces learned knowledge, preventing it from being overwritten by new information.
*   **No Need to Store Old Data:** Traditional replay methods require storing all historical data, which is prohibitively expensive. MyGO only needs to store lightweight generators, achieving massive storage compression.
*   **Knowledge Integration, Not Replacement:** The knowledge distillation process in the sleep phase enables the shared feature extractor to learn more general and robust representations that better serve all tasks.
*   **Biologically Inspired Interpretability:** The "wake-sleep" analogy provides an intuitive and logical framework for the complex process of continual learning.

<br>

---

## 🛠️ Quick Start

This section should include instructions on how to set up and run your project.

### 1. Prerequisites

*   Python 3.8+
*   PyTorch
*   TorchVision
*   NumPy

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/Cotix-AI/MyGO
cd MyGO

# Install dependencies
pip install torch torchvision numpy
```

### 3. Configuration

This project requires no additional configuration and can be run directly.

### 4. Running the Example

To launch the MyGO experiment and the comparative finetuning baseline, run the main script:

```bash
python main.py
```

You will see terminal output showing the learning performance after each task. The MyGO model will maintain high accuracy across all learned tasks, while the finetuning baseline's accuracy on old tasks will drop sharply, clearly demonstrating the problem of catastrophic forgetting.

<br>

---

## 🤝 How to Contribute

We welcome and encourage contributions to this project! If you have any ideas, suggestions, or find a bug, please feel free to submit a Pull Request or create an Issue.
