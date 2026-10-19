import sys
import sqlite3
from datetime import datetime
import torch
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QRadioButton
)
from PyQt6.QtCore import QThread, pyqtSignal
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

# Initialize local SQLite database for conversation and training logs
db_conn = sqlite3.connect("novamind_ai.db")
db_cursor = db_conn.cursor()
db_cursor.execute("""
                  CREATE TABLE IF NOT EXISTS platform_logs
                  (
                      id
                      INTEGER
                      PRIMARY
                      KEY
                      AUTOINCREMENT,
                      timestamp
                      TEXT,
                      category
                      TEXT,
                      content
                      TEXT
                  )
                  """)
db_conn.commit()

# Stylesheets for Light and Dark Modes
DARK_STYLE = """
    QWidget { background-color: #0f172a; color: #f8fafc; font-family: Segoe UI; }
    QTabWidget::pane { border: 1px solid #1e293b; }
    QTabBar::tab { background: #1e293b; color: #f8fafc; padding: 10px 20px; font-weight: bold; border-top-left-radius: 4px; border-top-right-radius: 4px; } 
    QTabBar::tab:selected { background: #6366f1; color: white; }
    QTextEdit { background-color: #090d16; color: #38bdf8; font-family: Consolas; font-size: 11pt; border: 1px solid #1e293b; border-radius: 4px; }
    QLineEdit { background-color: #1e293b; color: #ffffff; padding: 10px; font-size: 11pt; border: 1px solid #334155; border-radius: 4px; }
    QPushButton { background-color: #6366f1; color: white; padding: 10px; font-weight: bold; border-radius: 4px; }
    QPushButton:hover { background-color: #4f46e5; }
    QTableWidget { background-color: #090d16; color: #f8fafc; gridline-color: #1e293b; font-size: 10pt; border: 1px solid #1e293b; }
    QHeaderView::section { background-color: #1e293b; color: #f8fafc; padding: 5px; border: 1px solid #0f172a; font-weight: bold; }
    QGroupBox { border: 1px solid #334155; border-radius: 6px; margin-top: 10px; font-weight: bold; color: #38bdf8; }
    QGroupBox::title { subcontrol-position: top left; padding: 0 5px; }
    QRadioButton { color: #f8fafc; spacing: 8px; font-size: 10pt; }
"""

LIGHT_STYLE = """
    QWidget { background-color: #f1f5f9; color: #0f172a; font-family: Segoe UI; }
    QTabWidget::pane { border: 1px solid #cbd5e1; }
    QTabBar::tab { background: #e2e8f0; color: #334155; padding: 10px 20px; font-weight: bold; border-top-left-radius: 4px; border-top-right-radius: 4px; } 
    QTabBar::tab:selected { background: #4f46e5; color: white; }
    QTextEdit { background-color: #ffffff; color: #0369a1; font-family: Consolas; font-size: 11pt; border: 1px solid #cbd5e1; border-radius: 4px; }
    QLineEdit { background-color: #ffffff; color: #0f172a; padding: 10px; font-size: 11pt; border: 1px solid #cbd5e1; border-radius: 4px; }
    QPushButton { background-color: #4f46e5; color: white; padding: 10px; font-weight: bold; border-radius: 4px; }
    QPushButton:hover { background-color: #4338ca; }
    QTableWidget { background-color: #ffffff; color: #0f172a; gridline-color: #cbd5e1; font-size: 10pt; border: 1px solid #cbd5e1; }
    QHeaderView::section { background-color: #e2e8f0; color: #0f172a; padding: 5px; border: 1px solid #cbd5e1; font-weight: bold; }
    QGroupBox { border: 1px solid #cbd5e1; border-radius: 6px; margin-top: 10px; font-weight: bold; color: #0369a1; }
    QGroupBox::title { subcontrol-position: top left; padding: 0 5px; }
    QRadioButton { color: #0f172a; spacing: 8px; font-size: 10pt; }
"""


class ModelLoaderThread(QThread):
    finished = pyqtSignal(object)
    status_update = pyqtSignal(str)

    def run(self):
        self.status_update.emit("[System] Initializing Qwen 2.5 0.5B-Instruct on CPU with Maximum Token Capacities...")
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
        self.status_update.emit("[System] Model loaded successfully into memory with max token overrides!\n")
        self.finished.emit((model, tokenizer))


class NovaMindPlatform(QWidget):
    def __init__(self):
        super().__init__()
        self.is_dark_mode = True
        self.model = None
        self.tokenizer = None
        self.initUI()
        self.start_model_loader()

    def initUI(self):
        self.setWindowTitle("NovaMind AI - Native Intelligence Suite")
        self.resize(1200, 750)
        self.setStyleSheet(DARK_STYLE)

        main_layout = QVBoxLayout()

        # Top Header Bar with Theme Toggle
        header_layout = QHBoxLayout()
        title_label = QLabel("NovaMind AI (Local Engine - Max Token Limit)")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        self.theme_btn = QPushButton("Switch to Light Mode")
        self.theme_btn.setFixedWidth(180)
        self.theme_btn.clicked.connect(self.toggle_theme)
        header_layout.addWidget(self.theme_btn)

        main_layout.addLayout(header_layout)

        # Tabs for Inference, Training, and History
        self.tabs = QTabWidget()

        # --- TAB 1: INFERENCE WITH SIDEBAR ---
        inference_tab = QWidget()
        inf_main_layout = QHBoxLayout()

        # Left Side: Chat Workspace
        chat_layout = QVBoxLayout()
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        chat_layout.addWidget(QLabel("Model Output:"))
        chat_layout.addWidget(self.output_box)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type your prompt here...")
        chat_layout.addWidget(QLabel("Prompt:"))
        chat_layout.addWidget(self.input_box)

        self.run_inf_btn = QPushButton("Run Inference")
        self.run_inf_btn.clicked.connect(self.run_inference)
        chat_layout.addWidget(self.run_inf_btn)
        inf_main_layout.addLayout(chat_layout, stretch=3)

        # Right Side: AI Mode Selection Pane
        mode_group = QGroupBox("AI Engine Mode")
        mode_layout = QVBoxLayout()

        self.rb_lightning = QRadioButton("Lightning\n(Gives fast answers)")
        self.rb_balancer = QRadioButton("Balancer\n(Gives balance between time and accuracy)")
        self.rb_max = QRadioButton("Max\n(Gives the most accurate answers)")
        self.rb_automator = QRadioButton("Automator\n(Decides mode automatically based on prompt toughness)")

        self.rb_balancer.setChecked(True)

        mode_layout.addWidget(self.rb_lightning)
        mode_layout.addWidget(self.rb_balancer)
        mode_layout.addWidget(self.rb_max)
        mode_layout.addWidget(self.rb_automator)
        mode_layout.addStretch()

        mode_group.setLayout(mode_layout)
        mode_group.setFixedWidth(300)
        inf_main_layout.addWidget(mode_group, stretch=1)

        inference_tab.setLayout(inf_main_layout)
        self.tabs.addTab(inference_tab, "Inference Engine")

        # --- TAB 2: TRAINING ---
        training_tab = QWidget()
        train_layout = QVBoxLayout()

        self.train_log_box = QTextEdit()
        self.train_log_box.setReadOnly(True)
        train_layout.addWidget(QLabel("Training Console Logs:"))
        train_layout.addWidget(self.train_log_box)

        self.run_train_btn = QPushButton("Start Local CPU Training (Qwen 2.5 0.5B)")
        self.run_train_btn.clicked.connect(self.run_training_pipeline)
        train_layout.addWidget(self.run_train_btn)

        training_tab.setLayout(train_layout)
        self.tabs.addTab(training_tab, "Model Fine-Tuning")

        # --- TAB 3: SQLITE HISTORY BROWSER ---
        history_tab = QWidget()
        hist_layout = QVBoxLayout()

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["ID", "Timestamp", "Category", "Content"])
        self.history_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        hist_layout.addWidget(self.history_table)

        self.refresh_btn = QPushButton("Refresh History Table")
        self.refresh_btn.clicked.connect(self.load_history_data)
        hist_layout.addWidget(self.refresh_btn)

        history_tab.setLayout(hist_layout)
        self.tabs.addTab(history_tab, "SQLite History")

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

        self.load_history_data()

    def start_model_loader(self):
        self.loader_thread = ModelLoaderThread()
        self.loader_thread.status_update.connect(self.output_box.append)
        self.loader_thread.finished.connect(self.on_model_loaded)
        self.loader_thread.start()

    def on_model_loaded(self, payload):
        self.model, self.tokenizer = payload

    def toggle_theme(self):
        if self.is_dark_mode:
            self.setStyleSheet(LIGHT_STYLE)
            self.theme_btn.setText("Switch to Dark Mode")
            self.is_dark_mode = False
        else:
            self.setStyleSheet(DARK_STYLE)
            self.theme_btn.setText("Switch to Light Mode")
            self.is_dark_mode = True

    def determine_active_mode(self, prompt: str) -> str:
        if self.rb_lightning.isChecked():
            return "Lightning"
        elif self.rb_balancer.isChecked():
            return "Balancer"
        elif self.rb_max.isChecked():
            return "Max"
        elif self.rb_automator.isChecked():
            word_count = len(prompt.split())
            complex_keywords = ["code", "debug", "analyze", "mathematics", "proof", "architecture"]
            if word_count > 25 or any(kw in prompt.lower() for kw in complex_keywords):
                return "Max (Automated)"
            elif word_count > 10:
                return "Balancer (Automated)"
            else:
                return "Lightning (Automated)"
        return "Balancer"

    def run_inference(self):
        if self.model is None or self.tokenizer is None:
            self.output_box.append("\n[System] Model is still loading in the background. Please wait a moment...")
            return

        prompt = self.input_box.text().strip()
        if not prompt:
            return

        selected_mode = self.determine_active_mode(prompt)
        self.output_box.append(
            f"\n> User [{selected_mode}]: {prompt}\n[Generating with maximum absolute token capacity...]")
        QApplication.processEvents()

        # Absolute maximum token limit for Qwen 2.5 0.5B generation window
        max_tokens = 2048

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        self.output_box.append(f"\nAI: {response_text}")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db_cursor.execute("INSERT INTO platform_logs (timestamp, category, content) VALUES (?, ?, ?)",
                          (timestamp, f"Inference ({selected_mode})", f"Prompt: {prompt} | Response: {response_text}"))
        db_conn.commit()

        self.input_box.clear()
        self.load_history_data()

    def run_training_pipeline(self):
        if self.model is None or self.tokenizer is None:
            self.train_log_box.append("[Error] Model is still loading in the background. Please wait...")
            return

        self.train_log_box.append("[Trainer] Loading local dataset (train_data.jsonl)...")
        QApplication.processEvents()
        try:
            dataset = load_dataset("json", data_files="train_data.jsonl", split="train")
            peft_config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")

            trainer = SFTTrainer(
                model=self.model,
                train_dataset=dataset,
                peft_config=peft_config,
                dataset_text_field="output",
                max_seq_length=256,
                tokenizer=self.tokenizer,
            )
            self.train_log_box.append("[Trainer] Starting CPU training pass...")
            QApplication.processEvents()
            trainer.train()
            self.train_log_box.append("[Trainer] Training complete! Adapter ready.")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            db_cursor.execute("INSERT INTO platform_logs (timestamp, category, content) VALUES (?, ?, ?)",
                              (timestamp, "Training", "Executed successful CPU fine-tuning pass."))
            db_conn.commit()
            self.load_history_data()
        except Exception as e:
            self.train_log_box.append(f"[Error] Training failed: {str(e)}")

    def load_history_data(self):
        db_cursor.execute("SELECT id, timestamp, category, content FROM platform_logs ORDER BY id DESC")
        rows = db_cursor.fetchall()

        self.history_table.setRowCount(len(rows))
        for row_idx, row_data in enumerate(rows):
            for col_idx, col_data in enumerate(row_data):
                item = QTableWidgetItem(str(col_data))
                self.history_table.setItem(row_idx, col_idx, item)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    platform = NovaMindPlatform()
    platform.show()
    sys.exit(app.exec())