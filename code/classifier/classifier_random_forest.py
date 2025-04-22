import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter, defaultdict
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        super(Dataset, self).__init__()
        self.data = np.array(data, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_dataset(filepath):
    with open(filepath, "r", encoding="UTF-8") as file:
        data = json.load(file)
    return data

def prepare_data(dataset, labelset):
    train_data, test_data = dataset
    train_labels, test_labels = labelset
    return train_data, train_labels, test_data, test_labels

def train_random_forest(train_data, train_labels, n_estimators=250, max_depth=None,
                        min_samples_split=10, min_samples_leaf=1, max_features="sqrt",
                        bootstrap=True, class_weight=None):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        class_weight=class_weight,
        random_state=42
    )
    model.fit(train_data, train_labels)
    return model

def tune_hyperparameters(train_data, train_labels):
    # param_grid = {
    #     'n_estimators': [100, 250, 450, 600, 1000],
    #     'max_depth': [None, 10, 20, 30, 50],
    #     'min_samples_split': [2, 5, 10, 20],
    #     'min_samples_leaf': [1, 2, 3, 5, 10],
    #     'max_features': ["sqrt", "log2", 0.5, 0.75],
    #     'bootstrap': [True, False],
    #     'class_weight': ["balanced", "balanced_subsample", None],
    #     'criterion': ["gini", "entropy"]
    # }

    param_grid = {
        'n_estimators': [100, 250, 450],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'max_features': ["sqrt", 0.5, 0.75],
        'bootstrap': [True, False],
        'class_weight': ["balanced", None],
        'criterion': ["gini", "entropy"]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )
    grid_search.fit(train_data, train_labels)

    print("\nBest Parameters Found:", grid_search.best_params_)
    return grid_search.best_estimator_

# def test(model, test_data, test_labels):
#     predictions = model.predict(test_data)
#     accuracy = accuracy_score(test_labels, predictions)
#     label_names = ["suitable", "disturbing", "irrelevant", "restricted"]
#     report = classification_report(test_labels, predictions, target_names=label_names, output_dict=False, zero_division=0)
#     print(f"Accuracy: {accuracy * 100:.2f}%")
#     print("Classification Report:\n", report)
#     return accuracy, report

def test(model, test_data, test_labels):
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    label_names = ["suitable", "disturbing", "irrelevant", "restricted"]

    report = classification_report(test_labels, predictions, target_names=label_names, output_dict=True,
                                   zero_division=0)

    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report (in %):\n")
    print(f"{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1-score':>10}")
    print("-" * 44)

    for label in label_names:
        scores = report[label]
        print(
            f"{label:<12} {scores['precision'] * 100:>9.2f}% {scores['recall'] * 100:>9.2f}% {scores['f1-score'] * 100:>9.2f}%")

    print("-" * 44)
    print(
        f"{'Macro Avg':<12} {report['macro avg']['precision'] * 100:>9.2f}% {report['macro avg']['recall'] * 100:>9.2f}% {report['macro avg']['f1-score'] * 100:>9.2f}%")
    print(
        f"{'Weighted Avg':<12} {report['weighted avg']['precision'] * 100:>9.2f}% {report['weighted avg']['recall'] * 100:>9.2f}% {report['weighted avg']['f1-score'] * 100:>9.2f}%")

    return accuracy, report

def apply_resampling(train_data, train_labels, method="smote"):
    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)

    if method == "smote":
        sampler = SMOTE(sampling_strategy="auto", random_state=42)
    elif method == "undersampling":
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError("Invalid resampling method. Choose 'smote' or 'undersampling'")

    X_resampled, y_resampled = sampler.fit_resample(train_data, train_labels)

    print(f"Class distribution after {method}:", Counter(y_resampled))
    return X_resampled, y_resampled

def train_model_single_fold(data, labels, fold_id=0, gridsearch=False):
    train_data, train_labels, test_data, test_labels = prepare_data(data[fold_id], labels[fold_id])

    if gridsearch:
        print("\nRunning GridSearchCV for Hyperparameter Tuning...")
        model = tune_hyperparameters(train_data, train_labels)
        model.fit(train_data, train_labels)
    else:
        model = train_random_forest(train_data,
                                    train_labels,
                                    n_estimators=250,
                                    max_depth=None,
                                    min_samples_split=10,
                                    min_samples_leaf=1,
                                    max_features="sqrt",
                                    bootstrap=True,
                                    class_weight=None)

    accuracy, report = test(model, test_data, test_labels)

    return accuracy, report

def train_model_stratified_folds(data, labels, n_estimators=250, max_depth=None,
                                 min_samples_split=10, min_samples_leaf=1, max_features="sqrt",
                                 bootstrap=True, class_weight=None):
    fold_accuracies = []

    for fold_idx, ((train_data, test_data), (train_labels, test_labels)) in enumerate(zip(data, labels)):
        print(f"\nTraining on fold {fold_idx + 1}...")

        model = train_random_forest(train_data,
                                    train_labels,
                                    n_estimators,
                                    max_depth,
                                    min_samples_split,
                                    min_samples_leaf,
                                    max_features,
                                    bootstrap,
                                    class_weight)

        accuracy, _ = test(model, test_data, test_labels)
        fold_accuracies.append(accuracy)

    print(f"\nAverage Accuracy across {len(data)} folds: {np.mean(fold_accuracies) * 100:.2f}%")
    return fold_accuracies

def train_model_with_best_params(data, labels, params_path="best_params_per_fold.json"):
    try:
        with open(params_path, "r") as f:
            best_params_per_fold = json.load(f)
    except FileNotFoundError:
        print(f"No best parameters file found at '{params_path}'. Using default parameters.")
        best_params_per_fold = {}

    fold_accuracies = []

    for fold_id, ((train_data, test_data), (train_labels, test_labels)) in enumerate(zip(data, labels)):

        fold_key = f"fold_{fold_id + 1}"
        best_params = best_params_per_fold.get(f"fold_{fold_id}", {})

        if best_params:
            print(f"Using best parameters for {fold_key}: {best_params}")
        else:
            print(f"No specific parameters found for {fold_key}. Using default settings.")

        model = RandomForestClassifier(**best_params)
        model.fit(train_data, train_labels)

        accuracy, _ = test(model, test_data, test_labels)
        fold_accuracies.append(accuracy)

    return fold_accuracies

def train_model_with_best_params_fold(data, labels, params_path="best_params_per_fold.json", fold_id=6):
    try:
        with open(params_path, "r") as f:
            best_params_per_fold = json.load(f)
    except FileNotFoundError:
        print(f"No best parameters file found at '{params_path}'. Using default parameters.")
        best_params_per_fold = {}

    train_data, train_labels, test_data, test_labels = prepare_data(data[fold_id-1], labels[fold_id-1])
    best_params = best_params_per_fold.get(f"fold_{fold_id}", {})

    if best_params:
        print(f"Using best parameters for {fold_id}: {best_params}")
    else:
        print(f"No specific parameters found for {fold_id}. Using default settings.")

    model = RandomForestClassifier(**best_params)
    model.fit(train_data, train_labels)

    accuracy, report = test(model, test_data, test_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy, report

def train_model_stratified_folds_gridsearch(data, labels, output_path="best_params_per_fold.json"):
    fold_accuracies = []
    fold_reports = []
    best_params_per_fold = {}

    for fold_id, ((train_data, test_data), (train_labels, test_labels)) in enumerate(zip(data, labels)):
        best_model = tune_hyperparameters(train_data, train_labels)

        best_params_per_fold[f"fold_{fold_id + 1}"] = best_model.get_params()

        accuracy, report = test(best_model, test_data, test_labels)
        fold_accuracies.append(accuracy)
        fold_reports.append(report)

    with open(output_path, "w") as file:
        json.dump(best_params_per_fold, file, indent=4)

    print(f"\nBest parameters for each fold have been saved to '{output_path}'")

    avg_accuracy = np.mean(fold_accuracies)
    return fold_accuracies, fold_reports, avg_accuracy

def process_label(label):
    labels = ["suitable", "disturbing", "irrelevant", "restricted"]
    return labels.index(label)

if __name__ == "__main__":
    N_ESTIMATORS = 250
    MAX_DEPTH_TREE = None
    MIN_SAMPLES_SPLIT = 10
    MIN_SAMPLES_LEAF = 1
    MAX_FEATURES = "sqrt"
    BOOTSTRAP = True
    CLASS_WEIGHT = None

    data = load_dataset("../data/stratified_folds/stratified_folds_12_features.json")[5]
    labels = load_dataset("../data/stratified_folds/stratified_folds_labels_12_features.json")[5]

    # accuracies = []
    # macro_precisions = []
    # macro_recalls = []
    # macro_f1s = []
    #
    # for fold_idx, ((X_train, X_test), (y_train, y_test)) in enumerate(zip(data, labels), 1):
    #     model = RandomForestClassifier()
    #     model.fit(X_train, y_train)
    #
    #     print(f"\n--- Fold {fold_idx} ---")
    #     accuracy, report_dict = test(model, X_test, y_test)
    #
    #     accuracies.append(accuracy)
    #     macro_precisions.append(report_dict["macro avg"]["precision"])
    #     macro_recalls.append(report_dict["macro avg"]["recall"])
    #     macro_f1s.append(report_dict["macro avg"]["f1-score"])
    #
    # print("\n=== 8-Fold Cross-Validation Summary ===")
    # print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    # print(f"Average Precision (macro): {np.mean(macro_precisions):.4f}")
    # print(f"Average Recall (macro): {np.mean(macro_recalls):.4f}")
    # print(f"Average F1-score (macro): {np.mean(macro_f1s):.4f}")

    # accuracies = []
    # macro_precisions = []
    # macro_recalls = []
    # macro_f1s = []
    #
    # for fold_idx, ((X_train, X_test), (y_train, y_test)) in enumerate(zip(data, labels), 1):
    #     print(f"\n--- Fold {fold_idx} ---")
    #     model = tune_hyperparameters(X_train, y_train)
    #     accuracy, report_dict = test(model, X_test, y_test)
    #
    #     accuracies.append(accuracy)
    #     macro_precisions.append(report_dict["macro avg"]["precision"])
    #     macro_recalls.append(report_dict["macro avg"]["recall"])
    #     macro_f1s.append(report_dict["macro avg"]["f1-score"])
    #
    # print("\n=== 8-Fold Cross-Validation Summary ===")
    # print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    # print(f"Average Precision (macro): {np.mean(macro_precisions):.4f}")
    # print(f"Average Recall (macro): {np.mean(macro_recalls):.4f}")
    # print(f"Average F1-score (macro): {np.mean(macro_f1s):.4f}")

    train_data = load_dataset("../data/vectorized_data/vectorized_data_12_features.json")
    data2 = load_dataset("../data/datasets/remaining_data.json")
    train_labels_raw = [d.get("classification_label") for d in data2]
    train_labels = [process_label(label) for label in train_labels_raw]
    test_data = load_dataset("../data/vectorized_data/unseen_classifier_set_12_features.json")
    data3 = load_dataset("../data/datasets/unseen_classifier_set.json")
    test_labels_raw = [d.get("classification_label") for d in data3]
    test_labels = [process_label(label) for label in test_labels_raw]
    # test_data = data[5][1]
    # test_labels = labels[5][1]

    # res = defaultdict(int)
    # for predicted_class in test_labels:
    #     label_mapping = {0: 'suitable', 1: 'disturbing', 2: 'irrelevant', 3: 'restricted'}
    #     predicted_label = label_mapping[predicted_class]
    #     res[predicted_label] += 1

    # for label in test_labels_raw:
    #     res[label] += 1

    # print(json.dumps(res, indent=4))
    model = RandomForestClassifier(n_estimators=100,
                                   min_samples_leaf=3,
                                   min_samples_split=10,
                                   max_depth=None,
                                   max_features=0.5,
                                   criterion='entropy',
                                   class_weight=None,
                                   bootstrap=True,
                                   random_state=42)

    model.fit(train_data, train_labels)
    accuracy, report = test(model, test_data, test_labels)

    print("\n-------------------------------------------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(report)
    # joblib.dump(model, "../models/random_forest_gridsearch.pkl")

    # model = tune_hyperparameters(train_data, train_labels)
    # train_data = data[0]
    # train_labels = labels[0]
    # test_data = data[1]
    # test_labels = labels[1]

    # train_model_with_best_params_fold(data, labels)

    # train_model_single_fold(data, labels, 0, False)

    # model = RandomForestClassifier(n_estimators=250,
    #                                min_samples_leaf=1,
    #                                min_samples_split=10,
    #                                max_depth=30,
    #                                max_features='sqrt',
    #                                criterion='gini',
    #                                class_weight=None,
    #                                bootstrap=True,
    #                                random_state=42)

    # train_model_stratified_folds(data,
    #                              labels,
    #                              N_ESTIMATORS,
    #                              MAX_DEPTH_TREE,
    #                              MIN_SAMPLES_SPLIT,
    #                              MIN_SAMPLES_LEAF,
    #                              MAX_FEATURES,
    #                              BOOTSTRAP,
    #                              CLASS_WEIGHT)

    # accuracies, reports, avg_accuracy = train_model_stratified_folds_gridsearch(data, labels)
    # print(f"\nAverage accuracy with GridSearchCV: {avg_accuracy * 100:.2f}%")
    # fold_accuracies = train_model_with_best_params(data, labels)

    # SMOTE (Oversampling)
    # X_train_smote, y_train_smote = apply_resampling(train_data, train_labels, method="smote")
    # model_smote = train_random_forest(X_train_smote, y_train_smote, N_ESTIMATORS, MAX_DEPTH_TREE)
    # print("\nEvaluation with SMOTE:")
    # test(model_smote, test_data, test_labels)
    #
    # # Undersampling
    # X_train_rus, y_train_rus = apply_resampling(train_data, train_labels, method="undersampling")
    # model_rus = train_random_forest(X_train_rus, y_train_rus, N_ESTIMATORS, MAX_DEPTH_TREE)
    # print("\nEvaluation with Undersampling:")
    # test(model_rus, test_data, test_labels)