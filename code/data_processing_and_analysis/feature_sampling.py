import json
import random
import numpy as np
from collections import Counter

def load_dataset(filepath):
    with open(filepath, "r", encoding="UTF-8") as file:
        data = json.load(file)
    return data

def save_to_file(data, filepath):
    with open(filepath, "w", encoding="UTF-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def extract_labels(data):
    return [entry.get("classification_label") for entry in data]

def balanced_sampling(data, labels):
    labels = np.array(labels)

    label_counts = Counter(labels)
    min_count = min(label_counts.values())

    print(f"Class distribution before balancing: {label_counts}")
    print(f"Sampling {min_count} instances per class")

    balanced_indices = []
    for label in label_counts.keys():
        class_indices = np.where(labels == label)[0]
        balanced_indices.extend(random.sample(class_indices.tolist(), min_count))

    balanced_data = [data[i] for i in balanced_indices]
    balanced_labels = [labels[i] for i in balanced_indices]

    print(f"Balanced dataset size: {len(balanced_data)}")
    return balanced_data, balanced_labels

if __name__ == "__main__":
    data_file = "../vectorized_data.json"
    labels_source_file = "datasets/remaining_data.json"
    output_data_file = "balanced_vectorized_data.json"
    output_labels_file = "balanced_vectorized_labels.json"

    data = load_dataset(data_file)
    original_data = load_dataset(labels_source_file)
    labels = extract_labels(original_data)
    print(len(labels))

    balanced_data, balanced_labels = balanced_sampling(data, labels)

    save_to_file(balanced_data, output_data_file)
    save_to_file(balanced_labels, output_labels_file)

    print(f"Balanced dataset saved to {output_data_file} and {output_labels_file}")
