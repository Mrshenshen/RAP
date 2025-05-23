import json
import numpy as np
import argparse
from datasets import load_dataset
import random

def load_epochs(file):
    epochs = []
    ids = set()
    current_epoch = []

    with open(file) as f:
        step_data = json.load(f)
        current_epoch.extend(step_data)
        ids.update(extract_question(sample['onlyid']) for sample in current_epoch)
        epochs.append(current_epoch)
        current_epoch = []       
            
    return epochs, ids


def extract_question(id):
    return str(id)

def calculate_accuracy(epoch_data, id_set):
    accuracies = {}
    for id in id_set:
        question = extract_question(id)
        if question:
            accuracies[question] = [0, 0]
    for sample in epoch_data:
        id = extract_question(sample['onlyid'])
        accuracies[id][1] += 1
        if sample['reward'] == 1:
            accuracies[id][0] += 1
    return {id: correct/total if total else -1 
            for id, (correct, total) in accuracies.items()}

def process_accuracy_sequences(id_accuracies, max_epochs):
    for accuracy_sequence in id_accuracies.values():
        for i in range(len(accuracy_sequence)-1):
            if accuracy_sequence[i] == -1 and accuracy_sequence[i+1] != -1:
                accuracy_sequence[i] = accuracy_sequence[i+1]
    valid_sequences = [(id, sequence) 
                      for id, sequence in id_accuracies.items() 
                      if -1 not in sequence[:max_epochs]]
    
    if not valid_sequences:
        return [], [], []
    
    ids, sequences = zip(*valid_sequences)
    sequences = [seq[:max_epochs] for seq in sequences]
    mean_sequence = np.mean(sequences, axis=0)
    return ids, sequences, mean_sequence

def calculate_similarity_score(sequence, baseline_sequence):
    squared_diff_sum = sum((acc - baseline)**2 for acc, baseline in zip(sequence, baseline_sequence))
    max_diff_sum = sum((1 - baseline)**2 for baseline in baseline_sequence)
    return 1 - squared_diff_sum/max_diff_sum


def parse_args():
    parser = argparse.ArgumentParser(description='Process training data and filter ids')
    parser.add_argument('--txt_json', type=str, default='txt.json',
                      help='Path to the txt json file')
    parser.add_argument('--image_json', type=str, default='image.json',
                      help='Path to the image json file')
    parser.add_argument('--dataset_path', type=str, default='/path/to/dataset',
                      help='Path to the dataset')
    parser.add_argument('--similarity_threshold', type=float, default=0.85,
                      help='Minimum similarity score threshold for selecting ids')
    return parser.parse_args()

def main():
    args = parse_args()
    
    epochs, ids = load_epochs(args.txt_json)
    epochs_image, ids_image = load_epochs(args.image_json)
    epoch_accuracies = [calculate_accuracy(epoch, ids) for epoch in epochs]
    epoch_accuracies_image = [calculate_accuracy(epoch, ids_image) for epoch in epochs_image]
    id_accuracies = {id: [epoch[extract_question(id)] for epoch in epoch_accuracies] 
                        for id in ids}
    id_accuracies_image = {id: [epoch[extract_question(id)] for epoch in epoch_accuracies_image] 
                        for id in ids_image}
    valid_ids, accuracy_sequences, baseline_sequence = process_accuracy_sequences(
        id_accuracies, 1)
    valid_ids_image, accuracy_sequences_image, baseline_sequence_image = process_accuracy_sequences(
        id_accuracies_image, 1)
    print(baseline_sequence_image)
    print(baseline_sequence)
    diff = [a - b for a, b in zip(baseline_sequence_image, baseline_sequence)]
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
    id_scores = {
        id: (calculate_similarity_score(sequence, baseline_sequence_image),sequence)
        for id, sequence in zip(valid_ids_image, accuracy_sequences_image)
    }
    selected_ids_9 = set() 
    selected_ids = set() 
    for id, score in id_scores.items(): 
        if score[0] >= 0.6:
            for p, acc in zip(valid_ids, accuracy_sequences):
                if p == id:
                    if score[1][0]-acc[0]>0.2:
                        selected_ids_9.add(id)   
        if score[0] >= args.similarity_threshold: 
            for p, acc in zip(valid_ids, accuracy_sequences):
                if p == id:
                    if score[1][0]-acc[0]>diff[0]:
                        selected_ids.add(id)   
                    elif id in right1_samples and score[1][0]-acc[0]>=0.2:
                        selected_ids.add(id)
        for p, acc in zip(valid_ids, accuracy_sequences):
            if p == id: 
                if id in right1_samples and score[1][0]-acc[0]>=0.6:
                    selected_ids.add(id)
        
    data_path = args.dataset_path
    df = load_dataset("parquet", data_dir=data_path, split="train")
    filtered_indices = [idx for idx, sample in enumerate(df) if extract_question(sample['onlyid']) in selected_ids_9 and extract_question(sample['onlyid']) not in easy_samples]
    inter_rignt1_samples = [idx for idx, sample in enumerate(df) if extract_question(sample['onlyid']) in selected_ids and extract_question(sample['onlyid']) in right1_samples]
    random.seed(42)
    sample_size = int(len(inter_rignt1_samples) * 0.1)
    sampled_data = set(random.sample(inter_rignt1_samples, sample_size))
    add_filtered_indices = list(sorted(set(filtered_indices).union(sampled_data)))

    
    pandas_df = df.to_pandas()
    filtered_df = pandas_df.iloc[add_filtered_indices]
    filtered_df.to_parquet("./parquet/train-00000-of-00001.parquet", index=False)
    filtered_df.to_parquet("./parquet/test-00000-of-00001.parquet", index=False)
    filtered_df.to_parquet("./parquet/validation-00000-of-00001.parquet", index=False)
if __name__ == "__main__":
    main()