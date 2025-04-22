import pandas as pd
import json

label_weights = {
    'suitable': 2.0,
    'irrelevant': 1.0,
    'restricted': -1.0,
    'disturbing': -2.0
}

def load_dataset(filepath):
    with open(filepath, "r+", encoding="UTF-8") as file:
        data = json.loads(file.read())
    return data

def prepare_data_dataframe(data):
    extracted_data = []

    for d in data:
        classification_label = d.get("classification_label")
        category_id = d.get("snippet", {}).get("categoryId", 0)
        view_count = d.get("statistics", {}).get("viewCount", 0)
        licensed_content = d.get("contentDetails", {}).get("licensedContent", False)
        if classification_label is not None and category_id is not None and view_count is not None and licensed_content is not None:
            extracted_data.append({
                "label": classification_label,
                "categoryId": int(category_id),
                "viewCount": int(view_count),
                "licensedContent": licensed_content
            })

    return extracted_data

def save_to_file(data, filepath):
    with open(filepath, "w+", encoding="UTF-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def create_categoryid_mappings(df, label_column='label', category_column='categoryId'):
    categoryid_scores = {}
    grouped = df.groupby([category_column, label_column]).size().unstack(fill_value=0)

    for category_id in grouped.index:
        total = grouped.loc[category_id].sum()
        if total == 0:
            continue

        probabilities = (grouped.loc[category_id] / total).to_dict()
        most_frequent_label = max(probabilities, key=probabilities.get)
        probability_score = probabilities[most_frequent_label] # * label_weights.get(most_frequent_label, 0.0)

        categoryid_scores[category_id] = {
            "most_frequent_label": most_frequent_label,
            "probability_score": probability_score
        }

    return categoryid_scores

def find_closest_viewcount(viewcount, viewcount_mappings):
    closest_value = min(viewcount_mappings.keys(), key=lambda x: abs(x - viewcount))
    return viewcount_mappings[closest_value]

def create_viewcount_mappings(df, label_column='label', viewcount_column='viewCount'):
    viewcount_scores = {}
    grouped = df.groupby([viewcount_column, label_column]).size().unstack(fill_value=0)

    for viewcount in grouped.index:
        total = grouped.loc[viewcount].sum()
        if total == 0:
            continue

        probabilities = (grouped.loc[viewcount] / total).to_dict()
        most_frequent_label = max(probabilities, key=probabilities.get)
        probability_score = probabilities[most_frequent_label] # * label_weights.get(most_frequent_label, 0.0)

        viewcount_scores[viewcount] = {
            "most_frequent_label": most_frequent_label,
            "probability_score": probability_score
        }

    return viewcount_scores

if __name__ == '__main__':
    dataset = load_dataset("../data/datasets/remaining_data.json")
    data = prepare_data_dataframe(dataset)
    df = pd.DataFrame(data)

    categoryid_scores = create_categoryid_mappings(df)
    viewcount_scores = create_viewcount_mappings(df)

    print("Category ID Scores:", categoryid_scores)
    print("ViewCount Scores:", viewcount_scores)

    save_to_file(categoryid_scores, "categoryid_scores.json")
    save_to_file(viewcount_scores, "viewcount_scores.json")