import sys
import json
import os
# os.environ['CUDA_VISIBLE_DEVICES']='3'

import fire
# import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModel, TextStreamer

# from utils.prompter import Prompter
import bitsandbytes as bnb
import warnings
warnings.filterwarnings('ignore')

print(torch.cuda.is_available())

load_8bit = False
base_model = 'meta-llama/Llama-2-7b-chat-hf'
model_path = 'meta-llama/Llama-2-7b-chat-hf'
model_name = 'LLaMA2'

if torch.cuda.is_available():
    device = "cuda"

if model_name in ['ziya', 'Qwen', 'LLaMA2']:
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

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)


import json
import os.path as osp
from typing import Union


class Prompter_LLaMA2(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, verbose: bool = False):
        super().__init__()

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
    ) -> str:        
        if input:
            prompt = instruction + input
        else:
            prompt = instruction
        system_message = "Please give a proper response to the instruction. Do not say 'I don't know."
        prompt_template=f'''[INST] <<SYS>>
{system_message}
<</SYS>>

{prompt} [/INST]'''
        
        return prompt_template

    def get_response(self, output: str) -> str:
        return output.split('[/INST]')[-1].strip(tokenizer.eos_token).strip()

prompter = Prompter_LLaMA2()

def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    max_new_tokens=4096,
    **kwargs,
):
    
    streamer = TextStreamer(tokenizer)
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
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            # streamer=streamer,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    return instruction, prompter.get_response(output)

instruction = 'hello'
instruction, response = evaluate(instruction)
print(response)

from recbole.utils.case_study import full_sort_topk
def get_instruction_ml(idx):
    # user_info
    user_info = df_user[df_user["user_id:token"] == int(idx)]
    user_age = user_info["age:token"].values[0]
    user_gender = 'male' if user_info["gender:token"].values[0] == 'M' else 'female'
    user_occupation = user_info["occupation:token"].values[0]

    # history_inter_info
    df_inter_user = df_inter[df_inter['user_id:token'] == int(idx)]
    his_id_list = df_inter_user['item_id:token'].tolist()[-50:] 
    uid_series = dataset.token2id(dataset.uid_field, [idx])
    topk_score, topk_iid_list = full_sort_topk(uid_series, rec_model, test_data, k=1, device=config['device'])
    rec_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu()) 

    # next_item_info
    next_item = df_item[df_item["item_id:token"] == int(rec_list)]
    next_item_title = next_item["movie_title:token_seq"].values[0]
    next_item_class = next_item["class:token_seq"].values[0]

    instruction = 'The history films watched by the customer are:\n'
    for i, id in enumerate(his_id_list):
        instruction += f'{i+1}: {df_item[df_item["item_id:token"] == id]["movie_title:token_seq"].values[0]}' + '\n'
        instruction += f'The class of the movie is {df_item[df_item["item_id:token"] == id]["class:token_seq"].values[0]}.\n'

    instruction += f"""
The age of the customer is {user_age}, the gender is {user_gender} and the customer's occupation is {user_occupation}. \
As a recommender system in the movie domain, based on the customer's historical viewing records, \
historical viewing movie information and user information, \
tell the customer why he or she needs to watch this movie, and what are the advantages of this movie.
Give reasons why the customer needs to watch this movie with the following title and class:
{next_item_title}.
The class of the movie is {next_item_class}.

Use ONLY the information mentioned above, especially the history record. \
You're responding to the CUSTOMER. Directly tell the customer the reason. \
Respond in SHORT and CONCISE language. Your response should be WITHIN 75 words.

"""
    return instruction

def get_instruction_Amazon(idx):
    # history_inter_info
    df_inter_user = df_inter[df_inter['user_id:token'] == idx]
    his_id_list = df_inter_user['item_id:token'].tolist()[-10:]
    uid_series = dataset.token2id(dataset.uid_field, [idx])
    topk_score, topk_iid_list = full_sort_topk(uid_series, rec_model, test_data, k=1, device=config['device'])
    rec_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu()) 
    print(rec_list)
    # next_item_info
    next_item = df_item[df_item["item_id:token"] == rec_list]
    next_item_title = next_item["title:token"].values[0]
    next_item_brand = next_item["brand:token"].values[0]
    brand_prompt = f'The brand of the movie is {next_item_brand}.\n' if isinstance(next_item_brand, str) else ''
    next_item_type = next_item["sales_type:token"].values[0]
    type_prompt = f'The type of the product is {next_item_type}.\n' if isinstance(next_item_type, str) else ''

    instruction = 'The history products bought by the customer are:\n'
    print(his_id_list)
    for i, id in enumerate(his_id_list):
        item = df_item[df_item["item_id:token"] == id]
        item_title = item["title:token"].values[0]
        item_brand = item["brand:token"].values[0]
        item_type = item["sales_type:token"].values[0]

        instruction += f'{i+1}: {item_title}' + '\n'
        instruction += f'The brand of the movie is {item_brand}.\n' if math.isnan(item_brand) else ''
        instruction += f'The type of the product is {item_type}.\n' if math.isnan(item_type) else ''

    instruction += f"""
The age of the customer is {user_age}, the gender is {user_gender} and the customer's occupation is {user_occupation}. \
As a recommender system in the movie domain, based on the customer's HISTORICAL viewing records, \
HISTORICAL viewing movie information and user information, \
tell the customer why he or she needs to watch this movie, and what are the advantages of this movie.
Give reasons why the customer needs to watch this movie with the following title and class:
{next_item_title}.
{brand_prompt}
{type_prompt}

Use ONLY the information mentioned above, especially the history record. \
You're responding to the CUSTOMER. Directly tell the customer the reason. \
Respond in SHORT and CONCISE language. Your response should be WITHIN 75 words.

"""
    return instruction

def get_instruction_mind(idx):
    # history_inter_info
    df_inter_user = df_inter[df_inter['user_id:token'] == int(idx)]
    his_id_list = df_inter_user['item_id:token'].tolist()[-10:]
    uid_series = dataset.token2id(dataset.uid_field, [idx])
    topk_score, topk_iid_list = full_sort_topk(uid_series, rec_model, test_data, k=1, device=config['device'])
    rec_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu()) 
    # next_item_info
    next_item = df_item[df_item["item_id:token"] == int(rec_list)]
    next_item_title = next_item["title:token_seq"].values[0]
    next_item_category = next_item["category:token_seq"].values[0]
    next_item_sub_category = next_item["sub_category:token_seq"].values[0]
    next_item_abstract = next_item["abstract:token_seq"].values[0]


    instruction = 'The history news viewed by the customer are:\n'
    for i, id in enumerate(his_id_list):
        item = df_item[df_item["item_id:token"] == id]
        item_title = item["title:token_seq"].values[0]
        item_category = item["category:token_seq"].values[0]
        item_sub_category = item["sub_category:token_seq"].values[0]

        instruction += f'{i+1}: {item_title}' + '\n'
        instruction += f'The category of the news is {item_category}.\n'
        instruction += f'The sub category of the news is {item_sub_category}.\n'

    instruction += f"""
As a recommender system in the news domin, according to the user's previous news viewing history , \
tell the customer why he or she needs to watch this news.
Give reasons why the customer needs to watch this news with the following title and category:
{next_item_title}.
The category of the news is {next_item_category}.
The sub category of the news is {next_item_sub_category}.

Use ONLY the information mentioned above, especially the history record. \
You're responding to the CUSTOMER. Directly tell the customer the reason. \
Respond in SHORT and CONCISE language. Your response should be WITHIN 75 words.

"""
    return instruction

def get_instruction_steam(idx):
    # history_inter_info
    df_inter_user = df_inter[df_inter['user_id:token'] == int(idx)]
    his_id_list = df_inter_user['product_id:token'].tolist()[-10:]
    uid_series = dataset.token2id(dataset.uid_field, [idx])
    topk_score, topk_iid_list = full_sort_topk(uid_series, rec_model, test_data, k=1, device=config['device'])
    rec_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu()) 
    # next_item_info
    next_item = df_item[df_item["id:token"] == int(rec_list)]
    next_item_title = next_item["app_name:token"].values[0]
    next_item_tag = next_item["genres:token_seq"].values[0]


    instruction = 'The history games played by the customer are:\n'
    for i, id in enumerate(his_id_list):
        item = df_item[df_item["id:token"] == id]
        item_title = item["app_name:token"].values[0]
        item_tag = item["genres:token_seq"].values[0]

        instruction += f'{i+1}: {item_title}' + '\n'
        instruction += f'The tags of the game are {item_tag}.\n'

    instruction += f"""
As a recommender system in the game playing domain, according to the user's HISTORICAL play record , \
tell the customer why he or she needs to play this game, and what are the advantages of this game.
Give reasons why the customer needs to play this game with the following title and tags:
{next_item_title}.
The tags of the game are {next_item_tag}.

Use ONLY the information mentioned above, especially the history record. \
You're responding to the CUSTOMER. Directly tell the customer the reason. \
Respond in SHORT and CONCISE language. Your response should be WITHIN 75 words.

"""
    return instruction

rec_model = 'LightGCN'
import pandas as pd
from recbole.quick_start import load_data_and_model

###########################################################################################
dataset_name = 'ml-100k'
rec_model_path = '../../RecBole/experiments/ml-100k/LightGCN/LightGCN-Aug-28-2023_11-31-16.pth'


config, rec_model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file=rec_model_path,
)

df_item = pd.read_csv(f'dataset/{dataset_name}/{dataset_name}.item', sep='\t')
df_inter = pd.read_csv(f'dataset/{dataset_name}/{dataset_name}.inter', sep='\t')
if 'ml' in dataset_name:
    df_user = pd.read_csv(f'dataset/{dataset_name}/{dataset_name}.user', sep='\t')

if dataset_name == 'ml-100k':
    get_instruction = get_instruction_ml
elif dataset_name == 'Amazon_All_beauty':
    get_instruction = get_instruction_Amazon
elif dataset_name == 'mind_small_dev':
    get_instruction = get_instruction_mind
elif dataset_name == 'steam':
    get_instruction = get_instruction_steam

from tqdm import tqdm
def get_valid_user_list(df):
    valid_user_list = []
    for id in tqdm(df['user_id:token'].unique().tolist()):
        if len(df_inter[df_inter['user_id:token'] == id]) >=5:
            valid_user_list.append(id)
        if len(valid_user_list) == 1000:
            break
    return valid_user_list
valid_user_list = get_valid_user_list(df_inter)

from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")

random.shuffle(valid_user_list)

result_save_path = f'../results/{rec_model}/LLaMA2/{dataset_name}/'
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)

df_result = pd.DataFrame(columns=['instruction', 'response'])
# i = 0
with open(result_save_path + 'results.csv', 'w') as f:
    for id in tqdm(valid_user_list[:1000]):
        instruction, response = evaluate(get_instruction(str(id)))
        df_result = df_result.append({'instruction': instruction, 'response': response}, ignore_index=True)
        # print(instruction)
        # print(response)
        # break
    df_result.to_csv(f, index=False)
f.close()


###########################################################################################
dataset_name = 'mind_small_dev'
rec_model_path = '../../RecBole/experiments/mind_small_dev/LightGCN/LightGCN-Aug-28-2023_19-46-06.pth'

config, rec_model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file=rec_model_path,
)

df_item = pd.read_csv(f'dataset/{dataset_name}/{dataset_name}.item', sep='\t')
df_inter = pd.read_csv(f'dataset/{dataset_name}/{dataset_name}.inter', sep='\t')
if 'ml' in dataset_name:
    df_user = pd.read_csv(f'dataset/{dataset_name}/{dataset_name}.user', sep='\t')

if dataset_name == 'ml-100k':
    get_instruction = get_instruction_ml
elif dataset_name == 'Amazon_All_beauty':
    get_instruction = get_instruction_Amazon
elif dataset_name == 'mind_small_dev':
    get_instruction = get_instruction_mind
elif dataset_name == 'steam':
    get_instruction = get_instruction_steam

from tqdm import tqdm
def get_valid_user_list(df):
    valid_user_list = []
    for id in tqdm(df['user_id:token'].unique().tolist()):
        if len(df_inter[df_inter['user_id:token'] == id]) >=5:
            valid_user_list.append(id)
        if len(valid_user_list) == 1000:
            break
    return valid_user_list
valid_user_list = get_valid_user_list(df_inter)

random.shuffle(valid_user_list)

result_save_path = f'../results/{rec_model}/LLaMA2/{dataset_name}/'
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)

df_result = pd.DataFrame(columns=['instruction', 'response'])
# i = 0
with open(result_save_path + 'results.csv', 'w') as f:
    for id in tqdm(valid_user_list[:1000]):
        instruction, response = evaluate(get_instruction(str(id)))
        df_result = df_result.append({'instruction': instruction, 'response': response}, ignore_index=True)
        # print(instruction)
        # print(response)
        # break
    df_result.to_csv(f, index=False)
f.close()

###########################################################################################
dataset_name = 'steam'
rec_model_path = '../../RecBole/experiments/steam/LightGCN/LightGCN-Aug-28-2023_12-02-29.pth'

config, rec_model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file=rec_model_path,
)

df_item = pd.read_csv(f'dataset/{dataset_name}/{dataset_name}.item', sep='\t')
df_inter = pd.read_csv(f'dataset/{dataset_name}/{dataset_name}.inter', sep='\t')
if 'ml' in dataset_name:
    df_user = pd.read_csv(f'dataset/{dataset_name}/{dataset_name}.user', sep='\t')

if dataset_name == 'ml-100k':
    get_instruction = get_instruction_ml
elif dataset_name == 'Amazon_All_beauty':
    get_instruction = get_instruction_Amazon
elif dataset_name == 'mind_small_dev':
    get_instruction = get_instruction_mind
elif dataset_name == 'steam':
    get_instruction = get_instruction_steam

from tqdm import tqdm
def get_valid_user_list(df):
    valid_user_list = []
    for id in tqdm(df['user_id:token'].unique().tolist()):
        if len(df_inter[df_inter['user_id:token'] == id]) >=5:
            valid_user_list.append(id)
        if len(valid_user_list) == 1000:
            break
    return valid_user_list
valid_user_list = get_valid_user_list(df_inter)

random.shuffle(valid_user_list)

result_save_path = f'../results/{rec_model}/LLaMA2/{dataset_name}/'
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)

df_result = pd.DataFrame(columns=['instruction', 'response'])
# i = 0
with open(result_save_path + 'results.csv', 'w') as f:
    for id in tqdm(valid_user_list[:1000]):
        instruction, response = evaluate(get_instruction(str(id)))
        df_result = df_result.append({'instruction': instruction, 'response': response}, ignore_index=True)
        # print(instruction)
        # print(response)
        # break
    df_result.to_csv(f, index=False)
f.close()