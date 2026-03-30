import random
import torch
import gc

import numpy as np

from transformers import TextIteratorStreamer
from functions import format_examples, standarize_response
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm 
from threading import Thread


######### GA functions #########

# Function to create each individual of the population
def create_individual(n_examples, k_examples):
    individual = np.zeros(n_examples, dtype=int)
    ones = np.random.choice(n_examples, k_examples, replace=False)
    individual[ones] = 1
    return individual

# Function to create the initial population
def create_population(n_examples, k_examples, pop_size):
    return [create_individual(n_examples, k_examples) for _ in range(pop_size)]

# Function to maintain K examples in childrens
def fix_individual(individual, k_examples):
    ones = np.where(individual == 1)[0]
    zeros = np.where(individual == 0)[0]

    if len(ones) > k_examples:
        to_zero = np.random.choice(ones, len(ones) - k_examples, replace=False)
        individual[to_zero] = 0
    elif len(ones) < k_examples:
        to_one = np.random.choice(zeros, k_examples - len(ones), replace=False)
        individual[to_one] = 1

    return individual

# Function to select 2 individual of the population
def selection(population, fitness):
    a, b = random.sample(population, 2)
    return a if fitness(a) > fitness(b) else b

# Function to cross 2 individuals
def crossover(parent1, parent2, n_examples, k_examples, cross_rate):
    if random.random() > cross_rate:
        return parent1.copy()

    mask = np.random.rand(n_examples) < 0.5
    child = np.where(mask, parent1, parent2)

    return fix_individual(child, k_examples)

# Function to mutate an individual
def mutate(individual, mut_rate):
    if random.random() < mut_rate:
        ones = np.where(individual == 1)[0]
        zeros = np.where(individual == 0)[0]

        if len(ones) > 0 and len(zeros) > 0:
            i = random.choice(ones)
            j = random.choice(zeros)
            individual[i], individual[j] = 0, 1

    return individual

######### Functions for calculating fitness #########

_fitness_cache = {}

# F1 score on test set (subset or full set)
def fitness_predict(individual, data_train, data_validation, initial_prompt, SYSTEM_PROMPT, models_with_system_role, model_path, correct_labels, model, tokenizer, terminators):
      
    key = tuple(individual)

    if key in _fitness_cache:
        return _fitness_cache[key]
    
    # Building prompt with selected examples
    individual_indices = np.where(individual == 1)[0]

    examples_dict = (
        data_train
        .iloc[individual_indices]
        .groupby("label")["text"]
        .apply(list)
        .to_dict()
    )

    formated_examples = format_examples(examples_dict)

    responses = []

    # Evaluation
    for _, row in tqdm(data_validation.iterrows(), total=len(data_validation)):

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

        if filtered_response == "incorrect-label":
            for label in correct_labels:
                if label != row['label']:
                    filtered_response = label
                    break

        responses.append(filtered_response)

        # Cleaning cache
        del inputs
        torch.cuda.empty_cache()
        gc.collect()

    precision, recall, f1, _ = precision_recall_fscore_support(
        data_validation['label'], responses, average="macro", zero_division=0
    )

    _fitness_cache[key] = f1

    return f1