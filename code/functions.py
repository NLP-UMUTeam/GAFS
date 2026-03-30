from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.cluster import KMeans

import torch.nn.functional as F
import numpy as np
import torch
import random
import re

SEED = 42

######### Auxiliary functions #########
def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_uncertainty(texts, model, tokenizer, batch_size):
    set_seed()
    uncertainties_list = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            uncertainties = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # Entropy
        uncertainties_list.append(uncertainties.cpu().numpy())
    return np.hstack(uncertainties_list) 

def standarize_response (response, correct_labels):
    
    if not isinstance(response, str):
        return "incorrect-label"

    filtered_response = response.strip().replace("\n", " ").replace("\t", " ").lower()

    if filtered_response in correct_labels:
        return filtered_response
    
    else:

        matches = [
            label for label in correct_labels
            if re.search(rf'\b{re.escape(label.lower())}\b', filtered_response)
        ]

        if len(matches) == 1:
            # Replace the closest match with the term from the specialised vocabulary
            return matches[0]
        else:
            # 0 matches or multiples -> invalid
            return "incorrect-label"

######### Functions to get embeddings with MarIA #########
def get_maria_embeddings(texts, model, tokenizer, batch_size):
    set_seed()
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings_list.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
    return np.vstack(embeddings_list)


######### Functions for sampling #########

def select_diverse_examples_normal_centroids(embeddings, k=5):
    set_seed()
    kmeans = KMeans(n_clusters=k, random_state=SEED)
    kmeans.fit(embeddings)
    cluster_centers = kmeans.cluster_centers_

    selected_indices = []
    for center in cluster_centers:
        distances = np.linalg.norm(embeddings - center, axis=1)
        closest_idx = np.argmin(distances)
        selected_indices.append(closest_idx)
    return selected_indices


# Obtains k examples that are both diverse and uncertain
def hybrid_sampling(embeddings, uncertainties, k, diversity_weight=0.7, uncertainty_weight=0.3):
    diverse_indices = select_diverse_examples_normal_centroids(embeddings, k=15)

    distances = np.linalg.norm(embeddings[diverse_indices] - np.mean(embeddings, axis=0), axis=1)
    diversity_scores  = distances / np.max(distances)
    
    combined_scores = diversity_weight * diversity_scores + uncertainty_weight * uncertainties[diverse_indices]
    uncertain_diverse_indices = np.argsort(-combined_scores)[:k]
    return [diverse_indices[i] for i in uncertain_diverse_indices]

######### Functions to get examples #########
def format_examples (examples):

    examples_prompt = """
    Examples {id} of texts labelled '{label}':
    {example}
    """

    text_examples = []

    for label in list(examples.keys()): 

        for idx, text in enumerate(examples[label]):

            subprompt = examples_prompt.format(id=idx, example=text, label=label)

            text_examples.append(subprompt)

    examples_subprompt = "\n".join(text_examples)

    return examples_subprompt

def get_random_examples (df, number_of_examples):

    labels = df['label'].unique()
    
    examples_dict = {}

    selected_indices = []

    for label in labels:
        label_df = df[df['label'] == label]

        sampled = label_df.sample(
            n=min(number_of_examples, len(label_df)),
            random_state=SEED
        )

        examples_dict[label] = sampled['text'].tolist()

        examples = label_df['text'].sample(n=min(number_of_examples, len(label_df)), random_state=SEED).tolist()
        examples_dict[label] = examples

        selected_indices.extend(sampled.index.tolist())

    return selected_indices, format_examples(examples_dict)


def get_clustered_examples_with_maria (df, data_path, number_of_examples):
    
    model_name = data_path + "model_finetuned_maria"
    
    if data_path.endswith("/homo-mex-2023/"):
        num_labels = 3
    else:
        num_labels = 2

    # Load models and tokenizer
    model = AutoModel.from_pretrained(model_name, device_map="auto")
    classification_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, device_map="auto")
    model.eval()
    classification_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    labels = df['label'].unique()

    embeddings_dict = {}
    examples_dict = {}

    selected_indices = []

    for label in labels:
        label_df = df[df['label'] == label]
        label_texts = label_df['text'].tolist()

        # Get embeddings and uncertainties
        label_embeddings = get_maria_embeddings(label_texts, model, tokenizer, batch_size=8)
        embeddings_dict[label] = label_embeddings
        label_uncertainties = get_uncertainty(label_texts, classification_model, tokenizer, batch_size=8)

        indices = hybrid_sampling(label_embeddings, label_uncertainties, k=number_of_examples) 
        examples = [label_df['text'].iloc[i] for i in indices]
        examples_dict[label] = examples

        selected_indices.extend(indices)
        
    return selected_indices, format_examples(examples_dict)