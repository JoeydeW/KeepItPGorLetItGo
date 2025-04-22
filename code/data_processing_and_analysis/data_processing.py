import json
import random
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold

from vectorizer import vectorize

def load_dataset(filepath):
    with open(filepath, "r+", encoding="UTF-8") as file:
        data = [json.loads(line) for line in file.readlines()]
    return data

def load_vectorized_data(filepath):
    with open(filepath, "r+", encoding="UTF-8") as file:
        data = json.loads(file.read())
    return data

# Save sampled sets
def save_to_file(data, filepath):
    with open(filepath, "w+", encoding="UTF-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def sample_recommender_unseen_set(data):
    recommender_set = []
    related_videos_set = []

    dataset_video_ids = {d.get("id") for d in data}

    for d in data:
        related_videos = d.get("relatedVideos")
        related_videos_dataset = [rv for rv in related_videos if rv in dataset_video_ids]

        if len(related_videos_dataset) >= 2:
            recommender_set.append(d)
            related_videos_set.extend(related_videos_dataset)

    return recommender_set, related_videos_set

def split_dataset(data, recommender_set, related_videos_set, set_size):
    recommender_ids = {v["id"] for v in recommender_set}
    remaining_data = [d for d in data if d.get("id") not in recommender_ids and d.get("id") not in related_videos_set]

    unseen_classifier_set = random.sample(remaining_data, set_size)

    remaining_data = [v for v in remaining_data if v not in unseen_classifier_set]

    return unseen_classifier_set, remaining_data

def process_label(label):
    labels = ["suitable", "disturbing", "irrelevant", "restricted"]
    return labels.index(label)

def stratified_kfold_split(vectorized_data, labels, n_splits):
    # labels = [d.get("classification_label") for d in data]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    labels_set = []

    for training_index, test_index in skf.split(vectorized_data, labels):
        training_set = [vectorized_data[i] for i in training_index]
        test_set = [vectorized_data[i] for i in test_index]
        folds.append((training_set, test_set))

        training_set_labels = [process_label(labels[i]) for i in training_index]
        test_set_labels = [process_label(labels[i]) for i in test_index]
        labels_set.append((training_set_labels, test_set_labels))

    return folds, labels_set

def prepare_data(dataset, n_splits=8):
    data = load_dataset(dataset)

    recommender_set, related_videos_set = sample_recommender_unseen_set(data)

    total_folds = n_splits + 1
    remaining_data_ids = {d.get("id") for d in data} - {d.get("id") for d in recommender_set} - set(related_videos_set)
    remaining_data = [d for d in data if d.get("id") in remaining_data_ids]

    # print(f"recommender_set: {len(recommender_set)}")
    # print(f"related_videos_set: {len(set(related_videos_set))}")
    # print(f"remaining_data: {len(remaining_data)}")

    set_size = len(remaining_data) // total_folds

    unseen_classifier_set, remaining_data = split_dataset(data, recommender_set, related_videos_set, set_size)

    leftover_data = len(remaining_data) % n_splits
    if leftover_data > 0:
        unseen_classifier_set.extend(random.sample(remaining_data, leftover_data))
        remaining_data = [v for v in remaining_data if v not in unseen_classifier_set]

    save_to_file(remaining_data, "datasets/remaining_data.json")
    # print(f"unseen_classifier_set: {len(unseen_classifier_set)}")
    # print(f"Remaining data size: {len(remaining_data)}")

    vectorized_data = vectorize.vectorize(remaining_data)
    stratified_folds, labels_set = stratified_kfold_split(vectorized_data, remaining_data, n_splits)

    save_to_file(recommender_set, "datasets/recommender_set.json")
    save_to_file(related_videos_set, "datasets/related_videos_set_ids.json")
    save_to_file(unseen_classifier_set, "datasets/unseen_classifier_set.json")
    save_to_file(stratified_folds, "stratified_folds/stratified_folds_all_features.json")
    save_to_file(labels_set, "../old/vectorized_chunks_old/stratified_folds_labels.json")

    return recommender_set, related_videos_set, unseen_classifier_set#, stratified_folds

if __name__ == "__main__":
    dataset = "vectorized_data_12_features.json"
    data = load_vectorized_data(dataset)

    dataset2 = "remaining_data.json"
    data2 = load_vectorized_data(dataset2)
    labels = [d.get("classification_label") for d in data2]

    kfolds, kfolds_labels = stratified_kfold_split(data, labels, 8)

    for i, kfold in enumerate(kfolds):
        print(f"Fold {i+1}: training set size {len(kfold[0])}, test set size {len(kfold[1])}")

    save_to_file(kfolds, "stratified_folds_12_features.json")
    save_to_file(kfolds_labels, "stratified_folds_labels_12_features.json")