import os
import sys
from typing import List

import fire
import wandb
import torch
import transformers
from datasets import load_dataset
import bitsandbytes as bnb
from transformers.generation.utils import GenerationConfig


"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    TaskType,
    prepare_model_for_kbit_training,
)
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModel,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed
)
from dataclasses import field, fields, dataclass
from utils.prompter import Prompter

# base_model：即基础的大模型，这里使用了Huggingface上的7B的半精度模型；
# data_path: 训练大模型所需的数据，每一条样例包含了instruction、input和output三个字段；
# output_dir：输出checkpoint的存储目录；
# batch_size：批次大小；
# micro_batch_size：由于采用了Gradient accumulation技术，这里展示的是实际上每个小step的批次大小，但是每个小的step不进行梯度回传，只有达到batch size时才回传一次；
# num_epochs：训练的epoch数；
# learning_rate：学习率；
# cutoff_len：截断长度；
# val_set_size：验证集大小，由于alpaca只给出了一个数据文件，所以这里从其中分割出2000条作为验证数据集；
# lora_r、lora_alpha、lora_dropout、lora_target_module：都是跟LoRA相关的参数，其中r代表了设置的秩的大小，lora_target_module则决定了要对哪些模块进行LoRA调优，一共有四个（k，q，v，o）

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

def get_lora_modules(base_model, model_name=None):
    print('Finding lora modules...')
    if model_name in ['Baichuan', 'ziya', 'LLaMA2', 'Qwen']:
        model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    else:
        model = AutoModel.from_pretrained(
            base_model,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit or 32-bit
        lora_module_names.remove('lm_head')

    del model
    for i in range(5):
        torch.cuda.empty_cache()
    return list(lora_module_names)

def train(
    # model/data params
    model_name: str = 'ChatGLM2',
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    val_set_size: int = 20,
    eval_steps: int = 32,
    save_steps: int = 32,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "query_key_value", 
        "dense", "dense_h_to_4h", 
        "dense_4h_to_h"
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = True,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "llama_med",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name_cn: str = "",  # The prompt template to use, will default to alpaca.
    prompt_template_name_en: str = "",  # The prompt template to use, will default to alpaca.

):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            # f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template cn: {prompt_template_name_cn}\n"
            f"prompt template en: {prompt_template_name_en}\n"
        )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir + '/train_args.txt', 'w') as f:
            print(
                f"Training Alpaca-LoRA model with params:\n"
                f"base_model: {base_model}\n"
                f"data_path: {data_path}\n"
                f"output_dir: {output_dir}\n"
                f"batch_size: {batch_size}\n"
                f"micro_batch_size: {micro_batch_size}\n"
                f"num_epochs: {num_epochs}\n"
                f"learning_rate: {learning_rate}\n"
                f"cutoff_len: {cutoff_len}\n"
                f"val_set_size: {val_set_size}\n"
                f"lora_r: {lora_r}\n"
                f"lora_alpha: {lora_alpha}\n"
                f"lora_dropout: {lora_dropout}\n"
                # f"lora_target_modules: {lora_target_modules}\n"
                f"train_on_inputs: {train_on_inputs}\n"
                f"add_eos_token: {add_eos_token}\n"
                f"group_by_length: {group_by_length}\n"
                f"wandb_project: {wandb_project}\n"
                f"wandb_run_name: {wandb_run_name}\n"
                f"wandb_watch: {wandb_watch}\n"
                f"wandb_log_model: {wandb_log_model}\n"
                f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
                f"prompt template cn: {prompt_template_name_cn}\n"
                f"prompt template en: {prompt_template_name_en}\n", file=f
            )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size # 梯度累计，即每个小step不进行梯度回传，只有达到batch size时才回传一次

    prompter_cn = Prompter(prompt_template_name_cn)
    prompter_en = Prompter(prompt_template_name_en)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1)) # GPU总数
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} # device_map是一个字典，key是空字符串，value是LOCAL_RANK，即GPU编号
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ, wandb是一个可视化工具，可以用来可视化训练过程
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if model_name == 'ChatGLM2' :
        print('Loading model...')
        model = AutoModel.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True
        )

        print('Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.padding_side = "left"  # Allow batched inference
    
        
    elif model_name == 'internlm-chat':
        lora_target_modules = get_lora_modules(base_model)
        print('Loading model...')
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True
        )

        print('Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.padding_side = "left"  # Allow batched inference
        
    
    elif model_name in ['Baichuan', 'ziya', 'LLaMA2', 'Qwen']:
        lora_target_modules = get_lora_modules(base_model, model_name)
        print('Loading model...')
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True
        )
        model.generation_config = GenerationConfig.from_pretrained(base_model)


        print('Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, trust_remote_code=True)
        if model_name in ['Qwen']:
            pass
        else:
            tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
            tokenizer.padding_side = "left"  # Allow batched inference

    print(f"lora_target_modules: {lora_target_modules}\n")


    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if is_contains_chinese(data_point["instruction"]):
            prompter = prompter_cn
        else:
            prompter = prompter_en
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model)

    # setup peft
    print('Loading Lora...')
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
    )

    model = get_peft_model(model, config)


    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=2023
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.1,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=8,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if val_set_size > 0 else None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=5,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
