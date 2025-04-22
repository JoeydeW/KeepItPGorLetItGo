import itertools
from collections import defaultdict

import numpy as np
import json
import joblib

from tqdm import tqdm

label_scores = {
    'suitable': 2.0,
    'irrelevant': 1.0,
    'unknown': 0.0,
    'restricted': -1.0,
    'disturbing': -2.0
}

categoryid_scores = {}
viewcount_scores = {}

def load_dataset(filepath):
    with open(filepath, "r+", encoding="UTF-8") as file:
        data = json.loads(file.read())
    return data

def save_to_file(data, filepath):
    with open(filepath, "w+", encoding="UTF-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def load_scoring_dicts():
    global categoryid_scores, viewcount_scores

    categoryid_scores = load_dataset("categoryid_scores.json")
    viewcount_scores = load_dataset("viewcount_scores.json")

def predict(model, video_features):
    if not isinstance(video_features, np.ndarray):
        video_features = np.array(video_features).reshape(1, -1)

    probabilities = model.predict_proba(video_features)[0]
    predicted_class = np.argmax(probabilities)
    confidence_score = probabilities[predicted_class]

    label_mapping = {0: 'suitable', 1: 'disturbing', 2: 'irrelevant', 3: 'restricted'}
    predicted_label = label_mapping[predicted_class]

    return predicted_label, confidence_score

def calculate_rerank_score_classifier(predicted_label, confidence):
    label_score = label_scores.get(predicted_label, 0.0)
    return label_score * confidence

def find_rerank_score_categoryid(category_id):
    category_data = categoryid_scores.get(str(category_id))
    if category_data:
        label = category_data["most_frequent_label"]
        label_score = label_scores.get(label, 0.0)
        probability_score = category_data["probability_score"]
        return label, label_score * probability_score
    else:
        return "unknown", 0.0

def find_rerank_score_viewcount(view_count):
    if str(view_count) in viewcount_scores:
        viewcount_data = viewcount_scores[str(view_count)]
        label = viewcount_data["most_frequent_label"]
        label_score = label_scores.get(label, 0.0)
        probability_score = viewcount_data["probability_score"]
        return label, label_score * probability_score

    closest_value = min(map(int, viewcount_scores.keys()), key=lambda x: abs(x - view_count))
    viewcount_data = viewcount_scores[str(closest_value)]
    label = viewcount_data["most_frequent_label"]
    label_score = label_scores.get(label, 0.0)
    probability_score = viewcount_data["probability_score"]
    return label, label_score * probability_score

def original_ranker(recommender_data, related_videos_data, output_file):
    ranked_results = {}

    for video in tqdm(recommender_data, unit="entry", total=len(recommender_data)):
        video_id = video.get("id")
        related_videos = video.get("relatedVideos", [])
        reranked_videos = []

        for related_video in related_videos:
            related_video_data = related_videos_data.get(related_video)

            if related_video_data is None:
                reranked_videos.append({
                    "video_id": related_video,
                    "groundtruth_label": "unknown"
                })
                continue

            groundtruth_label = related_video_data.get("classification_label")

            reranked_videos.append({
                "video_id": related_video,
                "groundtruth_label": groundtruth_label
            })

        ranked_results[video_id] = reranked_videos

    save_to_file(ranked_results, output_file)
    print(f"Ranked results saved to {output_file}")

    return ranked_results

def classifier_rerank(recommender_data, related_videos_data, vectorized_data, output_file, model):
    reranked_results = {}

    vectorized_vector_map = {}
    for entry in vectorized_data:
        for video_id, vector in entry.items():
            vectorized_vector_map[video_id] = vector

    for video in tqdm(recommender_data, unit="entry", total=len(recommender_data)):
        video_id = video.get("id")
        related_videos = video.get("relatedVideos", [])
        reranked_videos = []

        for i, related_video in enumerate(related_videos):
            features = vectorized_vector_map.get(related_video)

            if features is None:
                label = "unknown"
                final_score = label_scores[label]
                reranked_videos.append({
                    "video_id": related_video,
                    "groundtruth_label": label,
                    "final_score": final_score
                })
                continue

            groundtruth_label = related_videos_data.get(related_video).get("classification_label")
            predicted_label, confidence = predict(model, features)
            final_score = calculate_rerank_score_classifier(predicted_label, confidence)

            reranked_videos.append({
                "video_id": related_video,
                "groundtruth_label": groundtruth_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "final_score": final_score
            })

        if reranked_videos:
            reranked_videos.sort(key=lambda x: x["final_score"], reverse=True)

        reranked_results[video_id] = reranked_videos

    # for _, videos in reranked_results.items():
    #     df = pd.DataFrame(videos)
    #     df = df[["video_id", "predicted_label", "confidence", "final_score"]]
    #     print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))

    save_to_file(reranked_results, output_file)
    print(f"Reranked results saved to {output_file}")

    return reranked_results

def categoryid_rerank(recommender_data, related_videos_data, output_file):
    reranked_results = {}

    for video in tqdm(recommender_data, unit="entry", total=len(recommender_data)):
        video_id = video.get("id")
        related_videos = video.get("relatedVideos", [])
        reranked_videos = []

        for i, related_video in enumerate(related_videos):
            related_video_data = related_videos_data.get(related_video)

            if related_video_data is None:
                label = "unknown"
                final_score = label_scores[label]
                reranked_videos.append({
                    "video_id": related_video,
                    "groundtruth_label": label,
                    "final_score": final_score
                })
                continue

            groundtruth_label = related_videos_data.get(related_video).get("classification_label")
            category_id = int(related_video_data.get("snippet", {}).get("categoryId"))
            label, final_score = find_rerank_score_categoryid(category_id)

            reranked_videos.append({
                "video_id": related_video,
                "groundtruth_label": groundtruth_label,
                "predicted_label": label,
                "final_score": final_score
            })

        if reranked_videos:
            reranked_videos.sort(key=lambda x: x["final_score"], reverse=True)

        reranked_results[video_id] = reranked_videos

    save_to_file(reranked_results, output_file)
    print(f"Reranked results saved to {output_file}")

    return reranked_results

def viewcount_rerank(recommender_data, related_videos_data, output_file):
    reranked_results = {}

    for video in tqdm(recommender_data, unit="entry", total=len(recommender_data)):
        video_id = video.get("id")
        related_videos = video.get("relatedVideos", [])
        reranked_videos = []

        for i, related_video in enumerate(related_videos):
            related_video_data = related_videos_data.get(related_video)

            if related_video_data is None:
                label = "unknown"
                final_score = label_scores[label]
                reranked_videos.append({
                    "video_id": related_video,
                    "groundtruth_label": label,
                    "final_score": final_score
                })
                continue

            groundtruth_label = related_videos_data.get(related_video).get("classification_label")
            view_count = int(related_video_data.get("statistics", {}).get("viewCount", 0))
            label, final_score = find_rerank_score_viewcount(view_count)

            reranked_videos.append({
                "video_id": related_video,
                "groundtruth_label": groundtruth_label,
                "predicted_label": label,
                "final_score": final_score
            })

        if reranked_videos:
            reranked_videos.sort(key=lambda x: x["final_score"], reverse=True)

        reranked_results[video_id] = reranked_videos

    save_to_file(reranked_results, output_file)
    print(f"Reranked results saved to {output_file}")

    return reranked_results

def combmnz_fusion(reranked_lists, related_videos_data, output_file):
    min_max_values = defaultdict(lambda: defaultdict(lambda: {"min": float("inf"), "max": float("-inf"), "range": 1}))

    for list_name, video_rankings in reranked_lists.items():
        for video_id, related_videos in video_rankings.items():
            all_scores = [related_video["final_score"] for related_video in related_videos]

            if all_scores:
                min_score, max_score = min(all_scores), max(all_scores)
                score_range = max_score - min_score if max_score > min_score else 1

                min_max_values[video_id][list_name] = {
                    "min": min_score,
                    "max": max_score,
                    "range": score_range
                }

    normalized_scores = defaultdict(lambda: defaultdict(list))

    for list_name, video_rankings in reranked_lists.items():
        for video_id, related_videos in video_rankings.items():
            min_score = min_max_values[video_id][list_name]["min"]
            score_range = min_max_values[video_id][list_name]["range"]

            for related_video in related_videos:
                related_video_id = related_video["video_id"]
                normalized_score = (related_video["final_score"] - min_score) / score_range
                normalized_scores[video_id][related_video_id].append(normalized_score)

    fused_list = {}

    for video_id, related_videos in normalized_scores.items():
        document_scores = {}
        for related_video_id, scores in related_videos.items():
            related_video_data = related_videos_data.get(related_video_id)
            if related_video_data is None:
                groundtruth_label = "unknown"
            else:
                groundtruth_label = related_video_data.get("classification_label")
            freq_factor = sum(1 for score in scores if abs(score) > 0.0)
            sum_normalized_scores = sum(scores)
            combmnz_score = sum_normalized_scores * freq_factor

            document_scores[related_video_id] = {
                "groundtruth_label": groundtruth_label,
                "final_score": combmnz_score
            }

        fused_related_videos = [
            {
                "video_id": related_video_id,
                "groundtruth_label": data["groundtruth_label"],
                "final_score": data["final_score"]
            }
            for related_video_id, data in document_scores.items()
        ]
        fused_related_videos.sort(key=lambda x: x["final_score"], reverse=True)
        fused_list[video_id] = fused_related_videos

    save_to_file(fused_list, output_file)
    print(f"Reranked results saved to {output_file}")

    return fused_list

def combsum_fusion(reranked_lists, related_videos_data, output_file):
    min_max_values = defaultdict(lambda: defaultdict(lambda: {"min": float("inf"), "max": float("-inf"), "range": 1}))

    for list_name, video_rankings in reranked_lists.items():
        for video_id, related_videos in video_rankings.items():
            all_scores = [related_video["final_score"] for related_video in related_videos]

            if all_scores:
                min_score, max_score = min(all_scores), max(all_scores)
                score_range = max_score - min_score if max_score > min_score else 1

                min_max_values[video_id][list_name] = {
                    "min": min_score,
                    "max": max_score,
                    "range": score_range
                }

    normalized_scores = defaultdict(lambda: defaultdict(list))

    for list_name, video_rankings in reranked_lists.items():
        for video_id, related_videos in video_rankings.items():
            min_score = min_max_values[video_id][list_name]["min"]
            score_range = min_max_values[video_id][list_name]["range"]

            for related_video in related_videos:
                related_video_id = related_video["video_id"]
                normalized_score = (related_video["final_score"] - min_score) / score_range
                normalized_scores[video_id][related_video_id].append(normalized_score)

    fused_list = {}

    for video_id, related_videos in normalized_scores.items():
        document_scores = {}
        for related_video_id, scores in related_videos.items():
            related_video_data = related_videos_data.get(related_video_id)
            if related_video_data is None:
                groundtruth_label = "unknown"
            else:
                groundtruth_label = related_video_data.get("classification_label")

            combsum_score = sum(scores)

            document_scores[related_video_id] = {
                "groundtruth_label": groundtruth_label,
                "final_score": combsum_score
            }

        fused_related_videos = [
            {
                "video_id": related_video_id,
                "groundtruth_label": data["groundtruth_label"],
                "final_score": data["final_score"]
            }
            for related_video_id, data in document_scores.items()
        ]
        fused_related_videos.sort(key=lambda x: x["final_score"], reverse=True)
        fused_list[video_id] = fused_related_videos

    save_to_file(fused_list, output_file)
    print(f"Reranked results saved to {output_file}")

    return fused_list

def format_ranked_list(filepath, num_entries=8):
    with open(filepath, "r", encoding="UTF-8") as file:
        ranked_videos = json.load(file)

    for video_id, related_videos in itertools.islice(ranked_videos.items(), num_entries):
        print(f"\nReranked List for Video ID: {video_id}\n")
        for idx, video in enumerate(related_videos, start=1):
            if filepath == "../reranker/reranked_videos_random_forest_unknowns.json" or filepath == "../reranker/reranked_videos_random_forest_unknowns_irrelevant.json":
                print(f"{idx}. {video['predicted_label']}")
            else:
                print(f"{idx}. {video['groundtruth_label']}")
        print("\n" + "-"*40)

if __name__ == "__main__":
    load_scoring_dicts()
    recommender_data = load_dataset("../data/datasets/recommender_set.json")
    related_videos_set = load_dataset("../reranker/related_videos_map.json")
    related_videos_features = load_dataset("../data/vectorized_data/related_videos_set_12_features.json")
    model = joblib.load("../../results/classifier/random_forest_gridsearch.pkl")

    original_ranker(recommender_data, related_videos_set, output_file="ranked_videos.json")
    classifier_rerank(recommender_data, related_videos_set, related_videos_features, output_file="../../results/reranked_lists/reranked_videos_classifier.json", model=model)
    categoryid_rerank(recommender_data, related_videos_set, output_file="../../results/reranked_lists/reranked_videos_categoryid.json")
    viewcount_rerank(recommender_data, related_videos_set, output_file="../../results/reranked_lists/reranked_videos_viewcount.json")

    classifier_list = load_dataset("../../results/reranked_lists/reranked_videos_classifier.json")
    categoryid_list = load_dataset("../../results/reranked_lists/reranked_videos_categoryid.json")
    viewcount_list = load_dataset("../../results/reranked_lists/reranked_videos_viewcount.json")
    #
    reranked_lists = {
        "classifier_list": classifier_list,
        "categoryid_list": categoryid_list,
        "viewcount_list": viewcount_list
    }

    combmnz_fusion(reranked_lists, related_videos_set, output_file="../../results/reranked_lists/reranked_videos_combmnz.json")
    combsum_fusion(reranked_lists, related_videos_set, output_file="../../results/reranked_lists/reranked_videos_combsum.json")