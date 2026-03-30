from functions import set_seed, format_examples, standarize_response
from ga_functions import create_population, selection, crossover, mutate, fitness_predict
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from tqdm import tqdm 
from threading import Thread
from sklearn.model_selection import train_test_split

import time
import argparse
import csv
import torch
import gc

import numpy as np
import pandas as pd

set_seed()

start_time = time.perf_counter()

SYSTEM_PROMPT = "You are a classification model that is really good at following instructions. Please follow the user's instructions as precisely as possible."

# GA arguments
POP_SIZE = 10
N_GEN = 5
MUT_RATE = 0.1
CROSS_RATE = 0.8

K = 10
VALIDATION_PERCENTAGE = 0.1

# Script arguments
parser = argparse.ArgumentParser()

parser.add_argument('-m', type=int, help='Model to evaluate', required=True)
parser.add_argument('-d', type=str, help='Data path', required=True)
parser.add_argument('-t', type=int, help='Approach to evaluate each solution', default=0) # 0 Subset, 1 Full set

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
    fitness_approach = "evaluating on subset"

    data_train, data_validation = train_test_split(
        data_train,
        test_size=VALIDATION_PERCENTAGE,
        stratify=data_train["label"],
        random_state=42
    )

elif args.t == 1:
    fitness_approach = "evaluating on full dataset"

def run_ga(fitness):

    n_examples = len(data_train)

    # Creation of the initial population
    population = create_population(n_examples, K, POP_SIZE)
    best = None

    # For each gen
    for gen in range(N_GEN):

        # New population of the generation
        new_population = []

        # For each individual
        for _ in range(POP_SIZE):

            # Selection of individuals
            p1 = selection(population, fitness)
            p2 = selection(population, fitness)

            # Cross of the selected individuals
            child = crossover(p1, p2, n_examples, K, CROSS_RATE)

            # Mutation of the children
            child = mutate(child, MUT_RATE)

            # Adding new children to the population
            new_population.append(child)

        population = new_population

        best_gen = max(population, key=fitness)

        # Storing best results
        if best is None or fitness(best_gen) > fitness(best):
            best = best_gen

        print(f"Gen {gen}: best fitness = {fitness(best):.4f}")
        print("Selected examples:", np.where(best == 1)[0])

    return best

if __name__ == "__main__":


    print("Starting GAFS evaluation with model " + model_name + " using " + fitness_approach + " on dataset " + args.d)

    
    #### 1. Charging model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if model_path in {"meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"}:
        tokenizer.pad_token = tokenizer.eos_token

    terminators = [tokenizer.eos_token_id]

    # If specific models have different terminators, adjust as needed
    if model_path in {"meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"}:
        terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    #### 2. Obtaining few-shot prompt
    with open(args.d + "prompts/fs_prompt.txt", "r", encoding="utf-8") as f:
        initial_prompt = f.read()

    #### 2. Defining fitness function
    fitness = lambda ind: fitness_predict(
        ind,
        data_train=data_train,
        data_validation=data_validation,
        initial_prompt=initial_prompt,
        SYSTEM_PROMPT=SYSTEM_PROMPT, 
        models_with_system_role=models_with_system_role,
        model_path=model_path,
        correct_labels=correct_labels,
        model=model,
        tokenizer=tokenizer,
        terminators=terminators
    )

    #### 3. Running GA
    best_individual = run_ga(fitness)

    #### 4. Building prompt with selected examples
    examples_indices = np.where(best_individual == 1)[0]

    examples_dict = (
        data_train
        .iloc[examples_indices]
        .groupby("label")["text"]
        .apply(list)
        .to_dict()
    )

    formated_examples = format_examples(examples_dict)

    # Creating results csv path
    results_path = "results/ga_predict/" + args.d.split("/")[-2] + "/" + model_name

    if args.t == 0:

        results_path += "_subset"
        
    elif args.t == 1:
        
        results_path += "_full_set"

    # Saving indices of selected examples
    with open(results_path + "_examples.txt", "w") as f:
        for idx in examples_indices:
            f.write(f"{idx}\n")

    # Creating results csv
    with open(results_path + "_responses.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "text", "ground_truth", "model_response", "filtered_label"])

    #### 6. Evaluation
    for _, row in tqdm(data_test.iterrows(), total=len(data_test)):

        # Building the prompt with examples and text
        prompt = initial_prompt.format(examples=formated_examples, text=row['text'])

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

    print("End of the GAFS experimentation with " + model_name + " on dataset " + args.d)