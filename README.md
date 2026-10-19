# NovaMind AI 🧠⚡

NovaMind AI is a 100% native desktop artificial intelligence platform built from the ground up using **Python** and **PyQt6**. Designed specifically to run efficiently on low-resource, CPU-only hardware (such as an 8 GB RAM laptop), it brings local LLM inference, LoRA fine-tuning, and structured session persistence directly to your desktop without relying on cloud APIs.

---

## ✨ Core Features

* **Native PyQt6 Desktop Interface:** A fast, responsive multi-threaded GUI featuring dark and light mode themes, background model loaders to eliminate startup freezing, and an intuitive tabbed workspace.
* **Local Intelligence Engine:** Powered locally by Hugging Face Transformers utilizing `Qwen/Qwen2.5-0.5B-Instruct`, optimized for local CPU execution with official chat template integration for authentic assistant behavior.
* **Dynamic AI Modes:** Switch between four operational modes on the fly:
  * **Lightning:** Fast generation for quick answers.
  * **Balancer:** Optimal equilibrium between response speed and depth.
  * **Max:** Maximum token capacity and depth for detailed answers.
  * **Automator:** Dynamically assesses prompt complexity and length to pick the ideal execution mode.
* **Built-In Fine-Tuning Suite:** Train local LoRA adapters on custom JSONL datasets using `PEFT` and `TRL` (`SFTTrainer`) directly from the UI.
* **Persistent SQLite Logging:** Automatically records all user prompts, model responses, and training events into a local database (`novamind_ai.db`), viewable via the built-in SQLite History browser tab.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **GUI Framework:** PyQt6
* **AI / ML Frameworks:** PyTorch, Hugging Face Transformers, PEFT, TRL, Datasets, Accelerate
* **Database:** SQLite3

---

## 🚀 Getting Started

### Installation

1. Create .venv via "python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate"
2. Install dependecies "pip install -r requirements.txt"
3.Run via "python -u main.py"
4.File structure= "NovaMind_AI/
│
├── main.py              # Main application entry point, GUI, threads, and logic
├── requirements.txt     # Python package dependencies
├── train_data.jsonl     # Local training dataset for fine-tuning
├── novamind_ai.db       # SQLite database for system logs and inference history
└── README.md            # Project documentation"
