import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
import transformers
from transformers import Trainer
from datasets import load_dataset

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def build_instruction_prompt(instruction: str, tokenizer):
    messages = [
        {"role": "user", "content": instruction}
    ]
    model_inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if tokenizer.bos_token and model_inputs.startswith(tokenizer.bos_token):
        model_inputs = model_inputs[len(tokenizer.bos_token):]
    return model_inputs

def train_tokenize_function(example, tokenizer):

    instruction_tokens = tokenizer.encode(
        build_instruction_prompt(example['instruction'], tokenizer),
        max_length = tokenizer.model_max_length,
        truncation = True,
        add_special_tokens = True
    )

    if instruction_tokens[-1] == tokenizer.eos_token_id:
        instruction_tokens = instruction_tokens[:-1]

    response_tokens = tokenizer.encode(
        example['response'],
        max_length = tokenizer.model_max_length,
        truncation = True,
        add_special_tokens = False
    )

    response_tokens.append(tokenizer.eos_token_id)

    return dict(
        input_ids = (instruction_tokens + response_tokens)[:tokenizer.model_max_length],
        labels = ([torch.nn.CrossEntropyLoss().ignore_index] * len(instruction_tokens) + response_tokens)[:tokenizer.model_max_length]
    )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        max_len = max([len(x) for x in input_ids])

        input_ids = torch.tensor([
            x + [self.tokenizer.pad_token_id] * (max_len - len(x)) 
            for x in input_ids
        ])
        labels = torch.tensor([
            x + [torch.nn.CrossEntropyLoss().ignore_index] * (max_len - len(x)) 
            for x in labels
        ])
        
        return dict(
            input_ids = input_ids,
            labels = labels,
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id),
        )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default = None)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default = 4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.gradient_checkpointing_kwargs = {'use_reentrant' : False}
    training_args.save_only_model = True
    training_args.torch_empty_cache_steps = 1
    
    if training_args.local_rank == 0:
        print('='*100)
        print(training_args)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length = training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if training_args.local_rank == 0:
        print('='*100)
        print("Load tokenizer from {} over.".format(model_args.model_name_or_path))
        print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
        print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
        print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    # 修改配置并加载模型
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.use_cache = False
    config.attn_implementation = "flash_attention_2"
    config.max_position_embeddings = training_args.model_max_length if training_args.model_max_length > config.max_position_embeddings else config.max_position_embeddings
    config.torch_dtype = torch.bfloat16
    config.trust_remote_code = True

    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = model_args.model_name_or_path,
        config = config
    )

    if training_args.local_rank == 0:
        print('='*100)
        print("Load model from {} over.".format(model_args.model_name_or_path))
        print("Model config:\n", model.config)


    raw_train_datasets = load_dataset(
        'json',
        data_files=data_args.data_path,
        split="train",
        cache_dir=training_args.cache_dir
    )
    if training_args.local_rank > 0: 
        torch.distributed.barrier()
        
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        num_proc = 32,
        remove_columns = raw_train_datasets.column_names,
        load_from_cache_file = True, # not args.overwrite_cache
        desc = "Running Encoding",
        fn_kwargs = {"tokenizer" : tokenizer}
    )

    if training_args.local_rank == 0:
        torch.distributed.barrier()
        print("Training dataset samples:", len(train_dataset))

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(
        train_dataset = train_dataset, 
        eval_dataset = None, 
        data_collator = data_collator
    )

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    trainer.train()

    # trainer.save_model(training_args.output_dir + "/checkpoint-9999")
    trainer.save_state()

if __name__ == "__main__":
    train()