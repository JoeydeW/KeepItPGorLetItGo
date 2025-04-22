import json
import numpy as np

def load_dataset(filepath):
    with open(filepath, "r+", encoding="UTF-8") as file:
        data = json.loads(file.read())
    return data

def save_to_file(data, filepath):
    with open(filepath, "w+", encoding="UTF-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def discounted_cumulative_gain(scores):
    return sum([score / np.log2((idx + 1) + 1) for idx, score in enumerate(scores)])

def ndcg(ranked_list):
    k = len(ranked_list)
    if k == 0:
        return 0.0

    ideal_list = sorted(ranked_list, reverse=True)

    dcg_score = discounted_cumulative_gain(ranked_list[:k])
    idcg_score = discounted_cumulative_gain(ideal_list[:k])

    return dcg_score / idcg_score if idcg_score > 0 else 0.0

def reciprocal_rank(ranked_list, target_labels):
    for idx, label in enumerate(ranked_list):
        if label in target_labels:
            return 1 / (idx + 1)

    return 0.0

def average_precision(ranked_list, target_labels):
    hits = 0
    sum_precisions = 0.0

    for idx, label in enumerate(ranked_list):
        if label in target_labels:
            hits += 1
            sum_precisions += hits / (idx + 1)

    return sum_precisions / hits if hits > 0 else 0.0

def rec_inap_metric(ranked_list, bad_labels):
    N = len(ranked_list)
    if N == 0:
        return 0.0

    numerator = sum(
        (N - (idx + 1) + 1) * (label in bad_labels)
        for idx, label in enumerate(ranked_list)
    )

    normalization_factor = N * (N + 1) / 2

    return numerator / normalization_factor

def hit_at_1(ranked_list, target_labels):
    if ranked_list[0] in target_labels:
        return 1.0
    return 0.0

def evaluate_results(ranked_list):
    results = []

    ndcg_ap_scores = []
    ndcg_inap_scores = []
    mrr_ap_scores = []
    mrr_inap_scores = []
    map_ap_scores = []
    map_inap_scores = []
    hit_ap_scores = []
    hit_inap_scores = []
    rec_inap_scores = []

    ndcg_suitable_scores = []
    ndcg_irrelevant_scores = []
    ndcg_restricted_scores = []
    ndcg_disturbing_scores = []

    mrr_suitable_scores = []
    mrr_irrelevant_scores = []
    mrr_restricted_scores = []
    mrr_disturbing_scores = []

    map_suitable_scores = []
    map_irrelevant_scores = []
    map_restricted_scores = []
    map_disturbing_scores = []

    hit_suitable_scores = []
    hit_irrelevant_scores = []
    hit_restricted_scores = []
    hit_disturbing_scores = []

    for video_id, related_videos in ranked_list.items():
        groundtruth_labels = [video['groundtruth_label'] for video in related_videos]

        ndcg_ap_score = ndcg([1 if label in ['suitable', 'irrelevant'] else 0 for label in groundtruth_labels])
        ndcg_inap_score = ndcg([1 if label in ['restricted', 'disturbing'] else 0 for label in groundtruth_labels])
        rr_ap_score = reciprocal_rank(groundtruth_labels, ['suitable', 'irrelevant'])
        rr_inap_score = reciprocal_rank(groundtruth_labels, ['restricted', 'disturbing'])
        ap_ap_score = average_precision(groundtruth_labels, ['suitable', 'irrelevant'])
        ap_inap_score = average_precision(groundtruth_labels, ['restricted', 'disturbing'])
        hit_ap_score = hit_at_1(groundtruth_labels, ['suitable', 'irrelevant'])
        hit_inap_score = hit_at_1(groundtruth_labels, ['restricted', 'disturbing'])
        rec_inap = rec_inap_metric(groundtruth_labels, ['restricted', 'disturbing'])

        ndcg_suitable = ndcg([1 if label == 'suitable' else 0 for label in groundtruth_labels])
        ndcg_irrelevant = ndcg([1 if label == 'irrelevant' else 0 for label in groundtruth_labels])
        ndcg_restricted = ndcg([1 if label == 'restricted' else 0 for label in groundtruth_labels])
        ndcg_disturbing = ndcg([1 if label == 'disturbing' else 0 for label in groundtruth_labels])

        rr_suitable = reciprocal_rank(groundtruth_labels, ['suitable'])
        rr_irrelevant = reciprocal_rank(groundtruth_labels, ['irrelevant'])
        rr_restricted = reciprocal_rank(groundtruth_labels, ['restricted'])
        rr_disturbing = reciprocal_rank(groundtruth_labels, ['disturbing'])

        ap_suitable = average_precision(groundtruth_labels, ['suitable'])
        ap_irrelevant = average_precision(groundtruth_labels, ['irrelevant'])
        ap_restricted = average_precision(groundtruth_labels, ['restricted'])
        ap_disturbing = average_precision(groundtruth_labels, ['disturbing'])

        hit_suitable = hit_at_1(groundtruth_labels, ['suitable'])
        hit_irrelevant = hit_at_1(groundtruth_labels, ['irrelevant'])
        hit_restricted = hit_at_1(groundtruth_labels, ['restricted'])
        hit_disturbing = hit_at_1(groundtruth_labels, ['disturbing'])

        results.append({
            'video_id': video_id,
            'nDCG_appropriate': ndcg_ap_score,
            'nDCG_inappropriate': ndcg_inap_score,
            'RR_appropriate': rr_ap_score,
            'RR_inappropriate': rr_inap_score,
            'AP_appropriate': ap_ap_score,
            'AP_inappropriate': ap_inap_score,
            'HIT_appropriate': hit_ap_score,
            'HIT_inappropriate': hit_inap_score,
            'rec_inap': rec_inap,
            'nDCG_suitable': ndcg_suitable,
            'nDCG_irrelevant': ndcg_irrelevant,
            'nDCG_restricted': ndcg_restricted,
            'nDCG_disturbing': ndcg_disturbing,
            'RR_suitable': rr_suitable,
            'RR_irrelevant': rr_irrelevant,
            'RR_restricted': rr_restricted,
            'RR_disturbing': rr_disturbing,
            'AP_suitable': ap_suitable,
            'AP_irrelevant': ap_irrelevant,
            'AP_restricted': ap_restricted,
            'AP_disturbing': ap_disturbing,
            'HIT_suitable': hit_suitable,
            'HIT_irrelevant': hit_irrelevant,
            'HIT_restricted': hit_restricted,
            'HIT_disturbing': hit_disturbing
        })

        ndcg_ap_scores.append(ndcg_ap_score)
        ndcg_inap_scores.append(ndcg_inap_score)
        mrr_ap_scores.append(rr_ap_score)
        mrr_inap_scores.append(rr_inap_score)
        map_ap_scores.append(ap_ap_score)
        map_inap_scores.append(ap_inap_score)
        hit_ap_scores.append(hit_ap_score)
        hit_inap_scores.append(hit_inap_score)
        rec_inap_scores.append(rec_inap)

        ndcg_suitable_scores.append(ndcg_suitable)
        ndcg_irrelevant_scores.append(ndcg_irrelevant)
        ndcg_restricted_scores.append(ndcg_restricted)
        ndcg_disturbing_scores.append(ndcg_disturbing)

        mrr_suitable_scores.append(rr_suitable)
        mrr_irrelevant_scores.append(rr_irrelevant)
        mrr_restricted_scores.append(rr_restricted)
        mrr_disturbing_scores.append(rr_disturbing)

        map_suitable_scores.append(ap_suitable)
        map_irrelevant_scores.append(ap_irrelevant)
        map_restricted_scores.append(ap_restricted)
        map_disturbing_scores.append(ap_disturbing)

        hit_suitable_scores.append(hit_suitable)
        hit_irrelevant_scores.append(hit_irrelevant)
        hit_restricted_scores.append(hit_restricted)
        hit_disturbing_scores.append(hit_disturbing)

    avg_results = {
        'Mean_nDCG_appropriate': np.mean(ndcg_ap_scores),
        'Mean_nDCG_inappropriate': np.mean(ndcg_inap_scores),
        'MRR_appropriate': np.mean(mrr_ap_scores),
        'MRR_inappropriate': np.mean(mrr_inap_scores),
        'MAP_appropriate': np.mean(map_ap_scores),
        'MAP_inappropriate': np.mean(map_inap_scores),
        'Mean_HIT_appropriate': np.mean(hit_ap_scores),
        'Mean_HIT_inappropriate': np.mean(hit_inap_scores),
        'REC-INAP': np.mean(rec_inap_scores),
        'Mean_nDCG_suitable': np.mean(ndcg_suitable_scores),
        'Mean_nDCG_irrelevant': np.mean(ndcg_irrelevant_scores),
        'Mean_nDCG_restricted': np.mean(ndcg_restricted_scores),
        'Mean_nDCG_disturbing': np.mean(ndcg_disturbing_scores),
        'MRR_suitable': np.mean(mrr_suitable_scores),
        'MRR_irrelevant': np.mean(mrr_irrelevant_scores),
        'MRR_restricted': np.mean(mrr_restricted_scores),
        'MRR_disturbing': np.mean(mrr_disturbing_scores),
        'MAP_suitable': np.mean(map_suitable_scores),
        'MAP_irrelevant': np.mean(map_irrelevant_scores),
        'MAP_restricted': np.mean(map_restricted_scores),
        'MAP_disturbing': np.mean(map_disturbing_scores),
        'Mean_HIT_suitable': np.mean(hit_suitable_scores),
        'Mean_HIT_irrelevant': np.mean(hit_irrelevant_scores),
        'Mean_HIT_restricted': np.mean(hit_restricted_scores),
        'Mean_HIT_disturbing': np.mean(hit_disturbing_scores)
    }

    return results, avg_results

if __name__ == '__main__':
    youtube = load_dataset("ranked_videos.json")
    classifier = load_dataset("../reranker/ranked_lists/reranked_videos_classifier.json")
    categoryid = load_dataset("../reranker/ranked_lists/reranked_videos_categoryid.json")
    viewcount = load_dataset("../reranker/ranked_lists/reranked_videos_viewcount.json")
    combmnz = load_dataset("../reranker/ranked_lists/reranked_videos_combmnz.json")
    combsum = load_dataset("../reranker/ranked_lists/reranked_videos_combsum.json")

    results_youtube, avg_youtube = evaluate_results(youtube)
    results_classifier, avg_classifier = evaluate_results(classifier)
    results_categoryid, avg_categoryid = evaluate_results(categoryid)
    results_viewcount, avg_viewcount = evaluate_results(viewcount)
    results_combmnz, avg_combmnz = evaluate_results(combmnz)
    results_combsum, avg_combsum = evaluate_results(combsum)

    print(f"avg result youtube: {avg_youtube}")
    print(f"avg result classifier: {avg_classifier}")
    print(f"avg result categoryid: {avg_categoryid}")
    print(f"avg result viewcount: {avg_viewcount}")
    print(f"avg result combmnz: {avg_combmnz}")
    print(f"avg result combsum: {avg_combsum}")

    save_to_file(results_youtube, "evaluation/evaluation_results_ranked_videos.json")
    save_to_file(results_classifier, "evaluation/evaluation_results_reranked_videos_classifier.json")
    save_to_file(results_categoryid, "evaluation/evaluation_results_reranked_videos_categoryid.json")
    save_to_file(results_viewcount, "evaluation/evaluation_results_reranked_videos_viewcount.json")
    save_to_file(results_combmnz, "evaluation/evaluation_results_reranked_videos_combmnz.json")
    save_to_file(results_combmnz, "evaluation/evaluation_results_reranked_videos_combsum.json")

    print("Evaluation completed and results saved to evaluation/")