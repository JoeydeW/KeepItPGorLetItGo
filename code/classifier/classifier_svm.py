import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

def load_dataset(filepath):
    with open(filepath, "r", encoding="UTF-8") as file:
        data = json.load(file)
    return data

def load_labels(filepath):
    with open(filepath, "r", encoding="UTF-8") as file:
        labels = json.load(file)
    return labels

def apply_pca(X, n_components=50):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca

def train_svm_with_gridsearch(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf']
    }

    grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_svm(X_train, y_train):
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return acc, report

if __name__ == "__main__":
    data = load_dataset("../data/stratified_folds/stratified_folds_12_features.json")
    labels = load_labels("../data/stratified_folds/stratified_folds_labels_12_features.json")

    model_names = ["SVM", "SVM + PCA", "SVM + GridSearch", "SVM + PCA + GridSearch"]
    metrics = {name: {"acc": [], "precision": [], "recall": [], "f1": []} for name in model_names}

    for fold_idx, ((X_train, X_test), (y_train, y_test)) in enumerate(zip(data, labels), 1):
        print(f"\n=== Fold {fold_idx} ===")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        clf = train_svm(X_train_scaled, y_train)
        acc, report = evaluate_model(clf, X_test_scaled, y_test)
        metrics["SVM"]["acc"].append(acc)
        metrics["SVM"]["precision"].append(report["macro avg"]["precision"])
        metrics["SVM"]["recall"].append(report["macro avg"]["recall"])
        metrics["SVM"]["f1"].append(report["macro avg"]["f1-score"])

        X_train_pca, pca = apply_pca(X_train, n_components=75)
        X_test_pca = pca.transform(StandardScaler().fit_transform(X_test))
        clf = train_svm(X_train_pca, y_train)
        acc, report = evaluate_model(clf, X_test_pca, y_test)
        metrics["SVM + PCA"]["acc"].append(acc)
        metrics["SVM + PCA"]["precision"].append(report["macro avg"]["precision"])
        metrics["SVM + PCA"]["recall"].append(report["macro avg"]["recall"])
        metrics["SVM + PCA"]["f1"].append(report["macro avg"]["f1-score"])

        clf = train_svm_with_gridsearch(X_train_scaled, y_train)
        acc, report = evaluate_model(clf, X_test_scaled, y_test)
        metrics["SVM + GridSearch"]["acc"].append(acc)
        metrics["SVM + GridSearch"]["precision"].append(report["macro avg"]["precision"])
        metrics["SVM + GridSearch"]["recall"].append(report["macro avg"]["recall"])
        metrics["SVM + GridSearch"]["f1"].append(report["macro avg"]["f1-score"])

        clf = train_svm_with_gridsearch(X_train_pca, y_train)
        acc, report = evaluate_model(clf, X_test_pca, y_test)
        metrics["SVM + PCA + GridSearch"]["acc"].append(acc)
        metrics["SVM + PCA + GridSearch"]["precision"].append(report["macro avg"]["precision"])
        metrics["SVM + PCA + GridSearch"]["recall"].append(report["macro avg"]["recall"])
        metrics["SVM + PCA + GridSearch"]["f1"].append(report["macro avg"]["f1-score"])

    print("\n=== Summary of Average Metrics Across Folds ===")
    for model, values in metrics.items():
        print(f"\n{model}:")
        print(f"  Accuracy : {np.mean(values['acc']):.4f}")
        print(f"  Precision: {np.mean(values['precision']):.4f}")
        print(f"  Recall   : {np.mean(values['recall']):.4f}")
        print(f"  F1-score : {np.mean(values['f1']):.4f}")