# OpenSyntheticCC

## Introduction
OpenSyntheticCC is a repository for fine-tuning language models on synthetic Chain-of-Thought (CoT) and code datasets. It provides scripts and configurations for distributed training, especially with DeepSpeed, and supports large-scale supervised fine-tuning.

## Features
- Fine-tuning on synthetic CoT and code datasets
- Distributed training with DeepSpeed and torchrun
- Customizable training parameters via shell scripts
- Data collation and tokenization for instruction-following tasks
- Example scripts for quick start

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/OpenSyntheticCC.git
   cd OpenSyntheticCC
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare your dataset
- Format: JSONL, each line should contain `instruction` and `response` fields.

### 2. Fine-tune a model
- Edit `sft.sh` to set your model path, data path, and output directory.
- Run the script:
  ```bash
  bash sft.sh
  ```
- The script uses `torchrun` and DeepSpeed for distributed training. Training parameters (batch size, learning rate, etc.) can be modified in `sft.sh`.

### 3. Custom Training
- You can also run `finetune.py` directly:
  ```bash
  python finetune.py --model_name_or_path <MODEL_PATH> --data_path <DATA_PATH> --output_dir <OUTPUT_DIR> ...
  ```
- See `sft.sh` for a full example of arguments.

## Distributed Training
- DeepSpeed configuration is provided in `deepspeed.json`.
- The script supports multi-node and multi-GPU training.

## File Overview

- `finetune.py`: Main training script for supervised fine-tuning.
- `sft.sh`: Example shell script for distributed training.
- `deepspeed.json`: DeepSpeed configuration for efficient large model training.
- `git.sh`: Helper script for quick git add/commit/push.
- `.gitignore`: Ignore logs, archives, and Java-related files.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.