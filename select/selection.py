import json
import numpy as np
import argparse
from datasets import load_dataset
import random

def load_epochs(file):
    epochs = []
    prompts = set()
    current_epoch = []

    with open(file) as f:
        step_data = json.load(f)
        current_epoch.extend(step_data)
        prompts.update(extract_question(sample['onlyid']) for sample in current_epoch)
        epochs.append(current_epoch)
        current_epoch = []       
            
    return epochs, prompts


def extract_question(prompt):
    return str(prompt)

def calculate_accuracy(epoch_data, prompt_set):
    accuracies = {}
    for prompt in prompt_set:
        question = extract_question(prompt)
        if question:
            accuracies[question] = [0, 0]
    for sample in epoch_data:
        prompt = extract_question(sample['onlyid'])
        accuracies[prompt][1] += 1
        if sample['reward'] == 1:
            accuracies[prompt][0] += 1
    return {prompt: correct/total if total else -1 
            for prompt, (correct, total) in accuracies.items()}

def process_accuracy_sequences(prompt_accuracies, max_epochs):
    # Forward fill missing values
    for accuracy_sequence in prompt_accuracies.values():
        for i in range(len(accuracy_sequence)-1):
            if accuracy_sequence[i] == -1 and accuracy_sequence[i+1] != -1:
                accuracy_sequence[i] = accuracy_sequence[i+1]
    
    # Filter valid sequences
    valid_sequences = [(prompt, sequence) 
                      for prompt, sequence in prompt_accuracies.items() 
                      if -1 not in sequence[:max_epochs]]
    
    if not valid_sequences:
        return [], [], []
    
    prompts, sequences = zip(*valid_sequences)
    sequences = [seq[:max_epochs] for seq in sequences]
    mean_sequence = np.mean(sequences, axis=0)
    return prompts, sequences, mean_sequence

def calculate_similarity_score(sequence, baseline_sequence):
    squared_diff_sum = sum((acc - baseline)**2 for acc, baseline in zip(sequence, baseline_sequence))
    max_diff_sum = sum((1 - baseline)**2 for baseline in baseline_sequence)
    return 1 - squared_diff_sum/max_diff_sum


def parse_args():
    parser = argparse.ArgumentParser(description='Process training data and filter prompts')
    parser.add_argument('--txt_json', type=str, default='txt.json',
                      help='Path to the txt json file')
    parser.add_argument('--image_json', type=str, default='image.json',
                      help='Path to the image json file')
    parser.add_argument('--dataset_path', type=str, default='/path/to/dataset',
                      help='Path to the dataset')
    parser.add_argument('--similarity_threshold', type=float, default=0.85,
                      help='Minimum similarity score threshold for selecting prompts')
    return parser.parse_args()

def main():
    args = parse_args()
    
    epochs, prompts = load_epochs(args.txt_json)
    epochs_image, prompts_image = load_epochs(args.image_json)
    epoch_accuracies = [calculate_accuracy(epoch, prompts) for epoch in epochs]
    epoch_accuracies_image = [calculate_accuracy(epoch, prompts_image) for epoch in epochs_image]
    prompt_accuracies = {prompt: [epoch[extract_question(prompt)] for epoch in epoch_accuracies] 
                        for prompt in prompts}
    prompt_accuracies_image = {prompt: [epoch[extract_question(prompt)] for epoch in epoch_accuracies_image] 
                        for prompt in prompts_image}
    valid_prompts, accuracy_sequences, baseline_sequence = process_accuracy_sequences(
        prompt_accuracies, 1)
    valid_prompts_image, accuracy_sequences_image, baseline_sequence_image = process_accuracy_sequences(
        prompt_accuracies_image, 1)
    print(baseline_sequence_image)
    print(baseline_sequence)
    diff = [a - b for a, b in zip(baseline_sequence_image, baseline_sequence)]
    #difficult
    with open(args.image_json, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON should contain a list of samples")
    right1_samples = []
    easy_samples = []
    for i in range(0, len(data), 5):
        group = data[i:i+5]
        count_reward = sum(1 for sample in group if sample.get('reward', 1) >= 0.9)
        if count_reward==5:
            easy_samples.append(extract_question(group[0]['onlyid']))
        elif count_reward==1:
            right1_samples.append(extract_question(group[0]['onlyid']))
    # Calculate similarity scores
    prompt_scores = {
        prompt: (calculate_similarity_score(sequence, baseline_sequence_image),sequence)
        for prompt, sequence in zip(valid_prompts_image, accuracy_sequences_image)
    }
    selected_prompts_9 = set() 
    selected_prompts = set() 
    for prompt, score in prompt_scores.items(): 
        if score[0] >= 0.6:
            for p, acc in zip(valid_prompts, accuracy_sequences):
                if p == prompt:
                    if score[1][0]-acc[0]>0.2:
                        selected_prompts_9.add(prompt)   

        if score[0] >= args.similarity_threshold: 
            for p, acc in zip(valid_prompts, accuracy_sequences):
                if p == prompt:
                    if score[1][0]-acc[0]>diff[0]:
                        selected_prompts.add(prompt)   
                    elif prompt in right1_samples and score[1][0]-acc[0]>=0.2:
                        selected_prompts.add(prompt)
        
    data_path = args.dataset_path
    df = load_dataset("parquet", data_dir=data_path, split="train")
    filtered_indices = [idx for idx, sample in enumerate(df) if extract_question(sample['onlyid']) in selected_prompts_9 and extract_question(sample['onlyid']) not in easy_samples]
    inter_rignt1_samples = [idx for idx, sample in enumerate(df) if extract_question(sample['onlyid']) in selected_prompts and extract_question(sample['onlyid']) in right1_samples]
    random.seed(42)
    sample_size = int(len(inter_rignt1_samples) * 0.1)
    sampled_data = set(random.sample(inter_rignt1_samples, sample_size))
    add_filtered_indices = list(sorted(set(filtered_indices).union(sampled_data)))

    
    pandas_df = df.to_pandas()
    filtered_df = pandas_df.iloc[add_filtered_indices]
    filtered_df.to_parquet("./parquet/train-00000-of-00001.parquet", index=False)
    filtered_df.to_parquet("./parquet/test-00000-of-00001.parquet", index=False)
    filtered_df.to_parquet("./parquet/validation-00000-of-00001.parquet", index=False)
###########
if __name__ == "__main__":
    main()