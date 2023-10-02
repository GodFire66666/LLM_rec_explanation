import sys
import json
import os

import fire
# import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModel

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

def load_instruction(instruct_dir):
    input_data = []
    with open(instruct_dir, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            d = json.loads(line)
            input_data.append(d)
    return input_data


def main(
    model_name: str = 'ChatGLM2',
    load_8bit: bool = False,
    base_model: str = "",
    # the infer data, if not exists, infer the default instructions in code
    instruct_dir: str = "",
    use_lora: bool = True,
    lora_weights: str = "tloen/alpaca-lora-7b",
    # The prompt template to use, will default to med_template.
    prompt_template_name_cn: str = "文修template中",  # The prompt template to use, will default to alpaca.
    prompt_template_name_en: str = "文修template英",
):
    print("*******************************************")
    print(base_model)
    print(instruct_dir)
    print("*******************************************")
    prompter_cn = Prompter(prompt_template_name_cn)
    prompter_en = Prompter(prompt_template_name_en)
    # tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if model_name in ['ChatGLM2', 'internlm-chat']:
        print('Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        print('Loading model...')
        model = AutoModel.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        if use_lora:
            print(f"using lora {lora_weights}")
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()

    elif model_name in ['Baichuan', 'ziya', 'Qwen', 'LLaMA2']:
        print('Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        print('Loading model...')
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        if use_lora:
            print(f"using lora {lora_weights}")
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        **kwargs,
    ):
        if is_contains_chinese(instruction):
            prompter = prompter_cn
        else:
            prompter = prompter_en
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        
        with torch.no_grad():
            if model_name in ['ChatGLM2', 'internlm-chat', 'Baichuan', 'ziya', 'Qwen', 'LLaMA2']:
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )
        s = generation_output.sequences[0]
        if model_name == 'Qwen':
            output = tokenizer.decode(s, skip_special_tokens=True)
        else:
            output = tokenizer.decode(s)
        return prompter.get_response(output)

    def infer_from_json(instruct_dir):
        input_data = load_instruction(instruct_dir)
        if use_lora:
            output_path = lora_weights 
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        else:
            output_path = base_model 
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        with open(output_path + '/results.txt', 'w') as f:
            for d in input_data:
                instruction = d["instruction"]
                input = d["input"]
                output = d["output"]
                print("###infering:###")
                model_output = evaluate(instruction, input)
                print("###instruction###")
                print(instruction)
                if input:
                    print("###input###")
                    print(input)
                print("###golden output###")
                print(output)
                print("###model output###")
                print(model_output)
                print()

                print("###infering###", file=f)
                print("###instruction###", file=f)
                print(instruction, file=f)
                print("###input###", file=f)
                if input:
                    print(input, file=f)
                    print("###golden output###", file=f)
                print(output, file=f)
                print("###model output###", file=f)
                print(model_output, file=f)
                print(file=f)
            f.close()

    if instruct_dir != "":
        infer_from_json(instruct_dir)
    else:
        for instruction in [
            "我感冒了，怎么治疗",
            "一个患有肝衰竭综合征的病人，除了常见的临床表现外，还有哪些特殊的体征？",
            "急性阑尾炎和缺血性心脏病的多发群体有何不同？",
            "小李最近出现了心动过速的症状，伴有轻度胸痛。体检发现P-R间期延长，伴有T波低平和ST段异常",
        ]:
            print("Instruction:", instruction)
            print("Response:", evaluate(instruction))
            print()


if __name__ == "__main__":
    fire.Fire(main)
