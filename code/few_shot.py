from transformers import AutoTokenizer, AutoModelForCausalLM,TextIteratorStreamer
from functions import set_seed, standarize_response, get_random_examples, get_clustered_examples_with_maria
from tqdm import tqdm 
from threading import Thread

import time
import torch
import argparse
import csv
import gc
import pandas as pd

set_seed()

start_time = time.perf_counter()

SYSTEM_PROMPT = "You are a classification model that is really good at following instructions. Please follow the user's instructions as precisely as possible."

# Script arguments
parser = argparse.ArgumentParser()

parser.add_argument('-m', type=int, help='Model to evaluate', required=True)
parser.add_argument('-d', type=str, help='Data path', required=True)
parser.add_argument('-e', type=int, help='Number of examples', default=5)
parser.add_argument('-t', type=int, help='Approach to obtain examples', default=0) # 0 random, 1 embeddings with maria

args = parser.parse_args()

# List of models that can be evaluated
if args.m == 0: # Gemma 2
    model_path = "google/gemma-2-2b-it"
    model_name = "gemma_2_2b_it"
elif args.m == 1:
    model_path = "google/gemma-2-9b-it"
    model_name = "gemma_2_9b_it"

elif args.m == 2: # Llama 3.x
    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    model_name = "llama_3_1_8b_it"
elif args.m == 3:
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    model_name = "llama_3_2_1b_it"
elif args.m == 4:
    model_path = "meta-llama/Llama-3.2-3B-Instruct"
    model_name = "llama_3_2_3b_it"

elif args.m == 5: # Qwen 2.5
    model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    model_name = "qwen_2_5_1b"
elif args.m == 6:
    model_path = "Qwen/Qwen2.5-3B-Instruct"
    model_name = "qwen_2_5_3b"
elif args.m == 7:
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    model_name = "qwen_2_5_7b"

elif args.m == 8: # Mistral
    model_path = "mistralai/Mistral-7B-Instruct-v0.3"
    model_name = "mistral_v3_it"

elif args.m == 9: # Phi 3.5
    model_path = "microsoft/Phi-3.5-mini-instruct"
    model_name = "phi_3_5_mini_it"

models_with_system_role = {
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",

    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",

    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3.5-mini-instruct"
}

# Spliting the data of the hate_speech dataset
if args.d.endswith("/hate_speech/"):

    dataset = pd.read_csv(args.d + "hateeval-spanish.csv")

    # Creating index
    dataset = dataset.reset_index(drop=True)
    dataset["id"] = dataset.index

    dataset = dataset[['id', 'tweet', '__split', 'label']]
   
    data_train = dataset[dataset['__split'] == 'train']
    data_test = dataset[dataset['__split'] == 'test']
else:
    data_train = pd.read_csv(args.d + "training.csv")

    data_test = pd.read_csv(args.d + "test.csv")

# Filtering and normalizing columns of the dataset being evaluated
# Also defining correct labels for each dataset
if args.d.endswith("/detests-dis/"):

    data_train = data_train[['id', 'text', 'stereotype']]
    data_test = data_test[['id', 'text', 'stereotype']]

    data_train.rename(columns={"stereotype": 'label'}, inplace=True)
    data_test.rename(columns={"stereotype": 'label'}, inplace=True)

    correct_labels = ["stereotype", "non-stereotype"]

    # Mapping binary codification to text labels in test data
    label_map = {0: "non-stereotype", 1: "stereotype"}
    data_test["label"]  = data_test["label"].map(label_map)

elif args.d.endswith("/exist/"):
    data_train = data_train[['id', 'text', 'value']]
    data_test = data_test[['id', 'text', 'value']]

    data_train.rename(columns={"value": 'label'}, inplace=True)
    data_test.rename(columns={"value": 'label'}, inplace=True)

    correct_labels = ["sexist", "non-sexist"]

elif args.d.endswith("/hate_speech/"):
    data_train = data_train[['id', 'tweet', 'label']]
    data_test = data_test[['id', 'tweet', 'label']]

    data_train.rename(columns={"tweet": "text"}, inplace=True)
    data_test.rename(columns={"tweet": "text"}, inplace=True)

    correct_labels = ["hatespeech", "non_hatespeech"]

else:
    data_train = data_train[['id', 'tweet', 'label']]
    data_test = data_test[['id', 'tweet', 'label']]

    data_train.rename(columns={"tweet": "text"}, inplace=True)
    data_test.rename(columns={"tweet": "text"}, inplace=True)

    correct_labels = ["P", "NP", "NONE"]

# Example selection approach
if args.t == 0:
    examples_approach = "random examples"
elif args.t == 1:
    examples_approach = "embeddings with maria"

if __name__ == '__main__':

    print("###########################")
    if args.t == 0:
        print("Starting few-shot evaluation with model " + model_name + " using " + examples_approach + " approach on dataset " + args.d)
    else:
        print("Starting few-shot evaluation with model " + model_name + " calculating " + examples_approach + " approach on dataset " + args.d)

    # Charging model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if model_path in {"meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"}:
        tokenizer.pad_token = tokenizer.eos_token

    terminators = [tokenizer.eos_token_id]

    # If specific models have different terminators, adjust as needed
    if model_path in {"meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"}:
        terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    # Obtaining few-shot examples
    if args.t == 0:
        selected_indices, examples = get_random_examples(data_train, args.e)
    elif args.t == 1:
        selected_indices, examples = get_clustered_examples_with_maria(data_train, args.d, args.e)
    
    # Obtaining few-shot prompt
    with open(args.d + "prompts/fs_prompt.txt", "r", encoding="utf-8") as f:
        initial_prompt = f.read()

    # Creating results csv path
    results_path = "results/fs/" + args.d.split("/")[-2] + "/" + model_name

    if args.t == 0:

        results_path += "_random_examples"
        
    elif args.t == 1:
        
        results_path += "_embeddings_maria_normal_centroids"


    # Saving indices of selected examples
    with open(results_path + "_examples.txt", "w") as f:
        for idx in selected_indices:
            f.write(f"{idx}\n")

    # Creating results csv
    with open(results_path + "_responses.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "text", "ground_truth", "model_response", "filtered_label"])

    # Evaluation
    for _, row in tqdm(data_test.iterrows(), total=len(data_test)):

        # Building the prompt with examples and text
        prompt = initial_prompt.format(examples=examples, text=row['text'])

        # Applying chat template to the prompt
        if model_path in models_with_system_role:

            conversation = [
                {
                    "role": "system", 
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user", 
                    "content": prompt
                },
            ]

        else:

            conversation = [
                {"role": "user", "content": SYSTEM_PROMPT + prompt},
            ]

        prompt = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True)

        # Tokenizing
        inputs = tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to(model.device)

        streamer = TextIteratorStreamer(tokenizer,
                                    skip_prompt=True,
                                    skip_special_tokens=True)
        
        # Adjusting generation parameters
        generate_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=4096,
            eos_token_id=terminators, 
            pad_token_id =tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1,
        )

        # Generating response
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        response = "".join(text for text in streamer)

        # Filtering response
        filtered_response = standarize_response(response, correct_labels)

        # Writing response
        with open(results_path + "_responses.csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([row["id"], row["text"], row['label'], response, filtered_response])

        # Cleaning cache
        del inputs
        torch.cuda.empty_cache()
        gc.collect()

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    with open(results_path + "_time.txt", "w") as f:
        f.write(f"Tiempo total de ejecución: {elapsed:.4f} segundos\n")
        
    print("End of the few-shot experimentation with " + model_name + " on dataset " + args.d)
    print("###########################")