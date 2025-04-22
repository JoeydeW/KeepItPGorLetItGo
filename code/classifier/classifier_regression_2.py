import json
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def load_dataset(filepath):
    with open(filepath, "r", encoding="UTF-8") as file:
        data = json.load(file)
    return data

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

def prepare_data(dataset, labelset):
    train_data, test_data = dataset
    train_labels, test_labels = labelset
    return train_data, train_labels, test_data, test_labels

# def tune_logistic_regression(train_data, train_labels):
#     scaler = StandardScaler()
#     train_data = scaler.fit_transform(train_data)
#
#     param_grid = {
#         'C': [0.01, 0.1, 1, 10, 100],
#         'solver': ['lbfgs', 'liblinear'],
#         'penalty': ['l2']
#     }
#
#     model = LogisticRegression(max_iter=1000, random_state=42)
#     grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=3)
#     grid_search.fit(train_data, train_labels)
#
#     print(f"Best Parameters: {grid_search.best_params_}")
#     best_model = grid_search.best_estimator_
#
#     return best_model, scaler

def train_logistic_regression(train_data, train_labels):
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)

    model = LogisticRegression(solver='lbfgs', max_iter=500, random_state=42)
    model.fit(train_data, train_labels)

    return model, scaler

# def test(model, scaler, test_data, test_labels):
#     test_data = scaler.transform(test_data)
#     predictions = model.predict(test_data)
#     accuracy = accuracy_score(test_labels, predictions)
#     print(f"Accuracy: {accuracy * 100:.2f}%")
#     return accuracy

def test(model, scaler, test_data, test_labels):
    test_data = scaler.transform(test_data)
    predictions = model.predict(test_data)

    report = classification_report(test_labels, predictions, output_dict=True, zero_division=0)

    accuracy = report["accuracy"]
    precision = report["macro avg"]["precision"]
    recall = report["macro avg"]["recall"]
    f1 = report["macro avg"]["f1-score"]

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    INPUT_SIZE = 114
    OUTPUT_SIZE = 4

    data = load_dataset("../data/stratified_folds/stratified_folds_12_features.json")
    labels = load_dataset("../data/stratified_folds/stratified_folds_labels_12_features.json")

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for fold_idx, (d, l) in enumerate(zip(data, labels), 1):
        print(f"\n=== Fold {fold_idx} ===")
        train_data, train_labels, test_data, test_labels = prepare_data(d, l)

        model, scaler = train_logistic_regression(train_data, train_labels)
        # model, scaler = tune_logistic_regression(train_data, train_labels)

        acc, prec, rec, f1 = test(model, scaler, test_data, test_labels)

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    print("\n=== 8-Fold Cross-Validation Summary ===")
    print(f"Average Accuracy : {np.mean(accuracies) * 100:.2f}%")
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Average Recall   : {np.mean(recalls):.4f}")
    print(f"Average F1-score : {np.mean(f1s):.4f}")

    # X_train_smote, y_train_smote = apply_resampling(train_data, train_labels, method="smote")
    # model_smote, scaler_smote = train_logistic_regression(X_train_smote, y_train_smote)
    # print("\nEvaluation with SMOTE:")
    # test(model_smote, scaler_smote, test_data, test_labels)
    #
    # X_train_rus, y_train_rus = apply_resampling(train_data, train_labels, method="undersampling")
    # model_rus, scaler_rus = train_logistic_regression(X_train_rus, y_train_rus)
    # print("\nEvaluation with Undersampling:")
    # test(model_rus, scaler_rus, test_data, test_labels)