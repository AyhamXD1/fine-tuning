# Overview
my first project in data science, fine tuning a gpt-2 model.
-------------------------------------------------------------
# Key Features
Custom Fine-Tuning Pipeline: Leverages transfer learning by fine-tuning a pre-trained GPT-2 model on a specialized story dataset, adapting its writing style, tone, and narrative structure.

Automated Data Processing & Dynamic Padding: Built with a custom PyTorch Dataset and DataCollatorForLanguageModeling to handle dynamic padding at runtime, maximizing GPU memory efficiency and accelerating training speeds.

Integrated Evaluation & Validation: Features an automated 80/20 train-validation split using train_test_split, tracking eval_loss across training steps to prevent overfitting and automatically saving the best-performing model checkpoint.

Controlled & Creative Text Generation: Configured with advanced decoding strategies—including Temperature Scaling (0.7–0.8), Top-K / Top-P (Nucleus) Sampling, and N-gram Repetition Penalties (no_repeat_ngram_size=2)—to ensure coherent, engaging, and non-repetitive story generation.

Mixed-Precision Acceleration (FP16): Utilizes half-precision floating-point arithmetic via CUDA GPU support to reduce memory footprint and enable faster training iterations.

Seamless Local Persistence: Implements complete export and loading mechanisms for both the fine-tuned model weights and tokenizer configurations via save_pretrained and from_pretrained.
-------------------------------------------------------------------
# Tech Stack 
Python — Primary programming language for pipeline construction and model training.

PyTorch — Deep learning framework used for tensor computations, model training, and CUDA acceleration.

Hugging Face Transformers — Framework providing the pre-trained GPT-2 architecture, Trainer API, TrainingArguments, and text generation pipelines.

Hugging Face Datasets & Tokenizers — Libraries used for tokenization (GPT2Tokenizer), subword encoding, and dynamic batch collation (DataCollatorForLanguageModeling).

Data Processing & Analytics
Pandas — Used for reading, cleaning, and extracting story data from local Excel/CSV files (.xlsx / .csv).

Scikit-Learn — Utilized for dataset splitting (train_test_split) into training and validation sets to monitor model performance.

Optimization & Hardware Acceleration
CUDA / Mixed Precision (FP16) — Leveraged via PyTorch to enable 16-bit floating-point precision training on NVIDIA GPUs, drastically reducing VRAM usage and boosting training speed.

Development & Storage
VS Code — Development environment used for writing, execution, and debugging.

Local Persistence (safetensors / PyTorch Binaries) — Export mechanism for saving fine-tuned model checkpoints and tokenizer configs locally via save_pretrained for fast inference.
