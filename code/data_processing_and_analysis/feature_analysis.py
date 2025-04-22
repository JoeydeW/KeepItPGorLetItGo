import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, chi2_contingency
from collections import Counter, defaultdict
import textwrap
from sklearn.metrics import mutual_info_score

def load_dataset(filepath):
    with open(filepath, "r", encoding="UTF-8") as file:
        return json.load(file)

def stats_licensedcontent(data):
    values = [int(d.get("contentDetails", {}).get("licensedContent", 0)) for d in data]
    mean, var, std = np.mean(values), np.var(values), np.std(values)
    print(f"Licensed Content mean: {mean}, var: {var}, std: {std}")
    return mean, var, std

def show_licensedcontent_distribution(data):
    distribution = defaultdict(lambda: defaultdict(int))
    for d in data:
        distribution[str(d.get("classification_label")).capitalize()][d.get("contentDetails", {}).get("licensedContent", False)] += 1

    df = pd.DataFrame.from_dict(distribution, orient="index").fillna(0)
    df.plot(kind="bar", stacked=False, edgecolor="black", color=["#4c72b0" if lc else "#dd8452" for lc in df.columns])
    plt.xticks(rotation=0)
    # plt.title("License Content Distribution by Label")
    plt.tight_layout()
    plt.show()
    return distribution

def stats_categoryid(data):
    categories = [map_category_id(str(d.get("snippet", {}).get("categoryId", ""))) for d in data]
    counts = Counter(categories)
    most_common_category, most_common_count = counts.most_common(1)[0] if counts else ("Unknown", 0)
    print(f"Most Common Category: {most_common_category}, most common count: {most_common_count}")
    return counts, most_common_category, most_common_count

def show_categoryid_distribution(data):
    distribution = defaultdict(lambda: defaultdict(int))
    for d in data:
        category = map_category_id(str(d.get("snippet", {}).get("categoryId", "Unknown")))
        distribution[d.get("classification_label")][category] += 1

    max_y_value = max([max(categories.values(), default=0) for categories in distribution.values()], default=0)

    colors = {"suitable": "blue", "disturbing": "orange", "restricted": "red", "irrelevant": "green"}

    for label, categories in distribution.items():
        sorted_categories = dict(sorted(categories.items()))
        categories_wrapped = [textwrap.fill(cat, width=10) for cat in sorted_categories.keys()]
        plt.figure(figsize=(12, 8))
        plt.bar(categories_wrapped, sorted_categories.values(), color=colors.get(label, "black"))
        plt.xticks(rotation=45, ha="right")
        plt.subplots_adjust(bottom=0.2)
        plt.ylim(0, max_y_value + 50)
        # plt.title(f"Category ID Distribution - {label.capitalize()}")
        # plt.xlabel("Video Category")
        plt.ylabel("Number of Videos")
        plt.tight_layout()
        plt.show()

    return distribution

def show_categoryid_distribution_stacked_numbers(data):
    distribution = defaultdict(lambda: defaultdict(int))
    for d in data:
        category = map_category_id(str(d.get("snippet", {}).get("categoryId", "Unknown")))
        label = d.get("classification_label")
        distribution[category][label] += 1

    sorted_categories = sorted(distribution.keys())
    labels = ["suitable", "disturbing", "restricted", "irrelevant"]
    # colors = {"suitable": "#0a70b0", "disturbing": "#f7951d", "restricted": "#9d1a1d", "irrelevant": "#115027"}
    colors = {"suitable": "#4c72b0", "disturbing": "#dd8452", "restricted": "#c44e52", "irrelevant": "#55a868"}

    counts_per_label = {label: [distribution[cat].get(label, 0) for cat in sorted_categories] for label in labels}

    # plt.figure(figsize=(8, 6))
    bottom = np.zeros(len(sorted_categories))

    for label in labels:
        plt.bar(sorted_categories, counts_per_label[label], edgecolor="black", label=label.capitalize(), color=colors[label], bottom=bottom)
        bottom += np.array(counts_per_label[label])

    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel("# of Videos")
    # plt.title("Category ID Distribution per Label")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return distribution

def show_categoryid_distribution_stacked_percentage(data):
    distribution = defaultdict(lambda: defaultdict(int))
    label_totals = defaultdict(int)
    for d in data:
        category = map_category_id(str(d.get("snippet", {}).get("categoryId", "Unknown")))
        label = d.get("classification_label")
        distribution[category][label] += 1
        label_totals[label] += 1

    sorted_categories = sorted(distribution.keys())
    labels = ["suitable", "disturbing", "restricted", "irrelevant"]
    colors = {"suitable": "#4c72b0", "disturbing": "#dd8452", "restricted": "#c44e52", "irrelevant": "#55a868"}
    # colors = {"suitable": "#0a70b0", "disturbing": "#f7951d", "restricted": "#9d1a1d", "irrelevant": "#115027"}
    # colors = {"suitable": "tab:blue", "disturbing": "tab:orange", "restricted": "tab:red", "irrelevant": "green"}

    counts_per_label = {label: [distribution[cat].get(label, 0) / sum(distribution[cat].values()) * 100 if sum(distribution[cat].values()) > 0 else 0 for cat in sorted_categories] for label in labels}

    # plt.figure(figsize=(8, 6))
    bottom = np.zeros(len(sorted_categories))

    for label in labels:
        plt.bar(sorted_categories, counts_per_label[label], edgecolor="black", label=label.capitalize(), color=colors[label], bottom=bottom)
        bottom += np.array(counts_per_label[label])

    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel("% of Videos")
    # plt.title("Category ID Distribution per Label")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)
    plt.tight_layout()
    plt.show()

    return distribution

def stats_default_audio_language(data):
    languages = [str(d.get("snippet", {}).get("defaultAudioLanguage", "")) for d in data]
    counts = Counter(languages)
    most_common_language, most_common_count = counts.most_common(1)[0] if counts else ("Unknown", 0)
    print(f"Default Audio Language: {most_common_language}, most common count: {most_common_count}")
    return counts, most_common_language, most_common_count

def show_default_audio_language_distribution(data):
    distribution = defaultdict(lambda: defaultdict(int))
    for d in data:
        lang = str(d.get("snippet", {}).get("defaultAudioLanguage", "Unknown"))
        distribution[d.get("classification_label")][lang] += 1

    max_y_value = max([max(languages.values(), default=0) for languages in distribution.values()], default=0)

    for label, languages in distribution.items():
        sorted_languages = dict(sorted(languages.items()))
        languages_wrapped = [textwrap.fill(cat, width=10) for cat in sorted_languages.keys()]
        plt.figure(figsize=(16, 8))
        plt.bar(languages_wrapped, sorted_languages.values())
        plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
        plt.ylim(0, max_y_value + 50)
        # plt.title(f"Default Audio Language Distribution - {label.capitalize()}")
        # plt.xlabel("Default Audio Language")
        # plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
    return distribution

def show_default_audio_language_distribution_stacked_percentage(data):
    distribution = defaultdict(lambda: defaultdict(int))
    language_totals = defaultdict(int)

    for d in data:
        lang = str(d.get("snippet", {}).get("defaultAudioLanguage", "Unknown"))
        label = d.get("classification_label")
        distribution[lang][label] += 1
        language_totals[lang] += 1

    sorted_languages = sorted(distribution.keys())
    labels = ["suitable", "disturbing", "restricted", "irrelevant"]
    # colors = {"suitable": "#0a70b0", "disturbing": "#f7951d", "restricted": "#9d1a1d", "irrelevant": "#115027"}
    colors = {"suitable": "#4c72b0", "disturbing": "#dd8452", "restricted": "#c44e52", "irrelevant": "#55a868"}

    counts_per_label = {
        label: [distribution[lang].get(label, 0) / language_totals[lang] * 100 if language_totals[lang] > 0 else 0 for
                lang in sorted_languages] for label in labels}

    plt.figure(figsize=(10.24, 7.2))
    bottom = np.zeros(len(sorted_languages))

    for label in labels:
        plt.bar(sorted_languages, counts_per_label[label], edgecolor="black", label=label.capitalize(), color=colors[label], bottom=bottom)
        bottom += np.array(counts_per_label[label])

    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel("% of Videos")
    # plt.title("Default Audio Language Distribution per Label")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)
    # plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    return distribution

def show_default_audio_language_distribution_stacked_numbers(data):
    distribution = defaultdict(lambda: defaultdict(int))

    for d in data:
        lang = str(d.get("snippet", {}).get("defaultAudioLanguage", "Unknown"))
        label = d.get("classification_label")
        distribution[lang][label] += 1

    sorted_languages = sorted(distribution.keys())
    labels = ["suitable", "disturbing", "restricted", "irrelevant"]
    colors = {"suitable": "#4c72b0", "disturbing": "#dd8452", "restricted": "#c44e52", "irrelevant": "#55a868"}

    counts_per_label = {label: [distribution[lang].get(label, 0) for lang in sorted_languages] for label in labels}

    plt.figure(figsize=(11.52, 4.8))
    bottom = np.zeros(len(sorted_languages))

    for label in labels:
        plt.bar(sorted_languages, counts_per_label[label], edgecolor="black", label=label.capitalize(), color=colors[label], bottom=bottom)
        bottom += np.array(counts_per_label[label])

    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel("# of Videos")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)
    plt.tight_layout()
    plt.show()

    return distribution

def stats_title_topicchange(data):
    topic_change_values = [float(d.get("titleTopicChange", {}).get("TOPICCHANGE", 0)) for d in data]
    topic_change_mean, topic_change_var, topic_change_std = np.mean(topic_change_values), np.var(topic_change_values), np.std(topic_change_values)
    topic_change_res = {"TOPICCHANGE": {"mean": topic_change_mean, "variance": topic_change_var, "std": topic_change_std}}
    print(f"Title topic change mean: {topic_change_mean}, var: {topic_change_var}, std: {topic_change_std}")

    same_topic_values = [float(d.get("titleTopicChange", {}).get("SAMETOPIC", 0)) for d in data]
    same_topic_mean, same_topic_var, same_topic_std = np.mean(same_topic_values), np.var(same_topic_values), np.std(same_topic_values)
    same_topic_res = {"SAMETOPIC": {"mean": same_topic_mean, "variance": same_topic_var, "std": same_topic_std}}
    print(f"Title same topic mean: {same_topic_mean}, var: {same_topic_var}, std: {same_topic_std}")

    return topic_change_res, same_topic_res

def stats_description_topicchange(data):
    topic_change_values = [float(d.get("descriptionTopicChange", {}).get("TOPICCHANGE", 0)) for d in data]
    topic_change_mean, topic_change_var, topic_change_std = np.mean(topic_change_values), np.var(topic_change_values), np.std(topic_change_values)
    topic_change_res = {"TOPICCHANGE": {"mean": topic_change_mean, "variance": topic_change_var, "std": topic_change_std}}
    print(f"Description topic change mean: {topic_change_mean}, var: {topic_change_var}, std: {topic_change_std}")

    same_topic_values = [float(d.get("descriptionTopicChange", {}).get("SAMETOPIC", 0)) for d in data]
    same_topic_mean, same_topic_var, same_topic_std = np.mean(same_topic_values), np.var(same_topic_values), np.std(same_topic_values)
    same_topic_res = {"SAMETOPIC": {"mean": same_topic_mean, "variance": same_topic_var, "std": same_topic_std}}
    print(f"Description same topic mean: {same_topic_mean}, var: {same_topic_var}, std: {same_topic_std}")

    return topic_change_res, same_topic_res

def show_topicchange_distribution(raw_data, features_data, feature_name, title):
    distribution = defaultdict(lambda: {"TOPICCHANGE": [], "SAMETOPIC": []})

    for raw_entry, feature_entry in zip(raw_data, features_data):
        label = raw_entry.get("classification_label", "Unknown")
        topic_change_value = float(feature_entry.get(feature_name, {}).get("TOPICCHANGE", 0))
        same_topic_value = float(feature_entry.get(feature_name, {}).get("SAMETOPIC", 0))

        distribution[label]["TOPICCHANGE"].append(topic_change_value)
        distribution[label]["SAMETOPIC"].append(same_topic_value)

    label_order = ["suitable", "irrelevant", "restricted", "disturbing"]
    label_titles = [label.capitalize() for label in label_order]

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.8), sharex=True, sharey=True)

    for ax, label, title_text in zip(axes.flat, label_order, label_titles):
        values = distribution.get(label, {"TOPICCHANGE": [], "SAMETOPIC": []})
        if values["TOPICCHANGE"]:
            sns.kdeplot(values["TOPICCHANGE"], label="TOPICCHANGE", fill=True, ax=ax, color="#4c72b0")
        if values["SAMETOPIC"]:
            sns.kdeplot(values["SAMETOPIC"], label="SAMETOPIC", fill=True, ax=ax, color="#dd8452")

        ax.set_title(title_text)
        ax.set_xlabel("Topic change score")
        ax.set_ylabel("Density")
        ax.legend(loc="upper center")

    # fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    # colors = {"suitable": "tab:blue", "disturbing": "tab:orange", "restricted": "tab:red", "irrelevant": "green"}

    # for label, values in distribution.items():
    #     # plt.figure(figsize=(10, 5))
    #     sns.kdeplot(values["TOPICCHANGE"], label="TOPICCHANGE", fill=True)
    #     sns.kdeplot(values["SAMETOPIC"], label="SAMETOPIC", fill=True)
    #     # plt.title(f"{title} Distribution by Label - {label.capitalize()}")
    #     plt.xlabel("Topic change score")
    #     plt.ylabel("Density")
    #     plt.legend(loc="upper center")
    #     plt.tight_layout()
    #     plt.show()

    return distribution

def stats_title_emotions(data):
    emotions = ["anger", "fear", "joy", "sadness", "disgust", "trust", "anticipation", "positive", "negative", "surprise"]
    emotion_stats = {}

    for emotion in emotions:
        values = [float(d.get("titleEmotions", {}).get(emotion, 0)) for d in data]
        emotion_stats[emotion] = {"mean": np.mean(values), "variance": np.var(values), "std": np.std(values)}
    print(f"Title emotions: {emotion_stats}")

    return emotion_stats

def stats_description_emotions(data):
    emotions = ["anger", "fear", "joy", "sadness", "disgust", "trust", "anticipation", "positive", "negative", "surprise"]
    emotion_stats = {}

    for emotion in emotions:
        values = [float(d.get("descriptionEmotions", {}).get(emotion, 0)) for d in data]
        emotion_stats[emotion] = {"mean": np.mean(values), "variance": np.var(values), "std": np.std(values)}
    print(f"Description emotions: {emotion_stats}")

    return emotion_stats

def show_emotions_distribution(raw_data, features_data, feature_name, title, use_median):
    emotions = ["anger", "fear", "joy", "sadness", "disgust", "trust", "anticipation", "positive", "negative", "surprise"]
    distribution = defaultdict(lambda: defaultdict(list))

    for raw_entry, feature_entry in zip(raw_data, features_data):
        label = raw_entry.get("classification_label", "Unknown")
        for emotion in emotions:
            value = float(feature_entry.get(feature_name, {}).get(emotion, 0))
            distribution[label][emotion].append(value)

    if use_median:
        emotion_matrix = {
            label: {emotion: np.median(values[emotion]) if values[emotion] else 0 for emotion in emotions}
            for label, values in distribution.items()
        }
    else:
        emotion_matrix = {
            label: {emotion: np.mean(values[emotion]) if values[emotion] else 0 for emotion in emotions}
            for label, values in distribution.items()
        }

    df = pd.DataFrame(emotion_matrix).T  # Transpose for better visualization

    # plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    # plt.title(f"{title} by Classification Label")
    plt.xticks(rotation=45)
    # plt.xlabel("Emotion")
    # plt.ylabel("Classification Label")
    plt.tight_layout()
    plt.show()

    return df

def stats_title_hard_words(data):
    values = [float(d.get("titleHardWords", 0)) for d in data]
    mean, var, std = np.mean(values), np.var(values), np.std(values)
    print(f"Title hard words mean: {mean}, var: {var}, std: {std}")
    return mean, var

def stats_description_hard_words(data):
    values = [float(d.get("descriptionHardWords", 0)) for d in data]
    mean, var, std = np.mean(values), np.var(values), np.std(values)
    print(f"Description hard words mean: {mean}, var: {var}, std: {std}")
    return mean, var

def show_hardwords_distribution(raw_data, features_data, feature_name, title):
    distribution = defaultdict(list)

    for raw_entry, feature_entry in zip(raw_data, features_data):
        label = raw_entry.get("classification_label", "Unknown")
        value = float(feature_entry.get(feature_name, 0))
        distribution[label].append(value)

    label_order = ["suitable", "irrelevant", "restricted", "disturbing"]
    label_titles = [label.capitalize() for label in label_order]
    # colors = {"suitable": "#0a70b0", "disturbing": "#f7951d", "restricted": "#9d1a1d", "irrelevant": "#115027"}
    # colors = {"suitable": "tab:blue", "disturbing": "tab:orange", "restricted": "tab:red", "irrelevant": "green"}
    colors = {"suitable": "#4c72b0", "disturbing": "#dd8452", "restricted": "#c44e52", "irrelevant": "#55a868"}

    # max_y_value = max([len(values) for values in distribution.values()], default=0)

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.8), sharex=True, sharey=True)

    for ax, label, title_text in zip(axes.flat, label_order, label_titles):
        values = distribution.get(label, [])
        if values:
            sns.kdeplot(values, ax=ax, fill=True, color=colors.get(label, "gray"), label=label.capitalize())
        ax.set_title(title_text)
        ax.set_xlabel(f"Proportion of {title.lower()}")
        ax.set_ylabel("Density")

    # fig.suptitle(f"{title} Distribution by Label", fontsize=16)
    plt.tight_layout()
    plt.show()

    # for label, values in distribution.items():
    #     # plt.figure(figsize=(10, 5))
    #     sns.kdeplot(values, label=label.capitalize(), color=colors.get(label, "gray"), fill=True)
    #     # plt.title(f"{title} - {label.capitalize()}")
    #     plt.xlabel(f"Proportion of {title}")
    #     plt.ylabel("Density")
    #     # plt.legend(title="Label")
    #     plt.tight_layout()
    #     plt.show()

    return distribution

def show_hardwords_distribution_stacked(raw_data, features_data, feature_name, title):
    distribution = defaultdict(list)

    for raw_entry, feature_entry in zip(raw_data, features_data):
        label = raw_entry.get("classification_label", "Unknown")
        value = float(feature_entry.get(feature_name, 0))
        distribution[label].append(value)

    # plt.figure(figsize=(10, 6))
    colors = {"suitable": "#0a70b0", "disturbing": "#f7951d", "restricted": "#9d1a1d", "irrelevant": "#115027"}
    colors = {"suitable": "tab:blue", "disturbing": "tab:orange", "restricted": "tab:red", "irrelevant": "green"}
    # colors = {"suitable": "#4c72b0", "disturbing": "#dd8452", "restricted": "#c44e52", "irrelevant": "#55a868"}

    for label, values in distribution.items():
        sns.kdeplot(values, label=label.capitalize(), fill=True, color=colors.get(label, "gray"), alpha=0.6)
        # sns.histplot(values, label=label.capitalize(), bins=30, element="poly", fill=True,
        #              color=colors.get(label, "gray"), alpha=0.6, stat="count")

    plt.xlabel(f"Proportion of {title}")
    plt.ylabel("Density")
    # plt.title(f"Distribution of Hard Words Proportion")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return distribution

def stats_statistics(data, feature_name):
    values = [int(d.get("statistics", {}).get(feature_name, 0)) for d in data]
    mean, var, std = np.mean(values), np.var(values), np.std(values)
    print(f"{feature_name} mean: {mean}, var: {var}, std: {std}")
    return mean, var

def show_statistics_distribution_cdf(data, feature_name, title):
    distribution = defaultdict(list)

    for d in data:
        count = int(d.get("statistics", {}).get(feature_name, 0))
        if count > 0:
            distribution[d.get("classification_label")].append(count)

    # plt.figure(figsize=(10, 5))
    colors = {"suitable": "blue", "disturbing": "orange", "restricted": "red", "irrelevant": "green"}
    # colors = {"suitable": "#4c72b0", "disturbing": "#dd8452", "restricted": "#c44e52", "irrelevant": "#55a868"}
    linestyles = {"suitable": "-", "disturbing": ":", "restricted": "--", "irrelevant": "-."}

    for label, counts in distribution.items():
        if not counts:
            continue

        sorted_counts = np.sort(counts)
        cumulative_probabilities = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)

        plt.plot(sorted_counts, cumulative_probabilities, color=colors.get(label, "black"),
                 linestyle=linestyles.get(label, "-"), label=label.capitalize())

    plt.xscale("log")
    plt.xlabel(f"# of {title}")
    plt.ylabel("CDF")
    # plt.title(f"CDF of {title}")
    plt.legend()
    # plt.grid(False, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    return distribution

def show_statistics_distribution(data, feature_name, title, color, y_addition):
    distribution = defaultdict(lambda: defaultdict(int))

    for d in data:
        count = int(d.get("statistics", {}).get(feature_name, 0))
        if count > 0:
            distribution[d.get("classification_label")][count] += 1

    max_y_value = max(
        [max(counts.values(), default=0) for counts in distribution.values()],
        default=0
    )

    colors = {"suitable": "#0a70b0", "disturbing": "#f7951d", "restricted": "#9d1a1d", "irrelevant": "#115027"}

    for label, counts in distribution.items():
        if not counts:
            continue

        count_values = list(counts.keys())
        frequencies = list(counts.values())

        bins = np.logspace(np.log10(min(count_values)), np.log10(max(count_values)), num=20)

        # plt.figure(figsize=(10, 5))
        plt.hist(count_values, bins=bins, weights=frequencies, edgecolor="black", alpha=0.7, color=colors.get(label, "black"))
        plt.xscale("log")
        plt.ylim(0, max_y_value + y_addition)
        # plt.xlim(0, 10**9)
        # plt.xlim(0, 10**7)
        # plt.xlim(0, 10**6)

        # plt.title(f"Histogram of {title} - {label.capitalize()} Videos")
        plt.xlabel(f"# of {title}")
        plt.ylabel("Number of Videos")
        plt.tight_layout()
        plt.show()

    return distribution

def stats_license(data):
    values = [int(d.get("status", {}).get("license") == "creativeCommon") for d in data]
    mean, var, std = np.mean(values), np.var(values), np.std(values)
    print(f"License mean: {mean}, var: {var}, std: {std}")
    return mean, var

def show_license_distribution(data):
    distribution = defaultdict(lambda: defaultdict(int))

    for d in data:
        license_status = d.get("status", {}).get("license", "creativeCommon")
        distribution[d.get("classification_label")][license_status] += 1

    # max_y_value = max(
    #     [max(license_counts.values(), default=0) for license_counts in distribution.values()],
    #     default=0
    # )
    #
    # license_types = ["creativeCommon", "youtube"]
    #
    # for license_type in license_types:
    #     # plt.figure(figsize=(8, 6))
    #
    #     labels = []
    #     values = []
    #     for label, license_counts in distribution.items():
    #         if license_counts.get(license_type, 0) > 0:
    #             labels.append(label)
    #             values.append(license_counts[license_type])
    #
    #     if not values:
    #         continue
    #
    #     plt.bar(labels, values, edgecolor="black", color="tab:blue" if license_type == "creativeCommon" else "tab:orange")
    #     # plt.ylim(0, max_y_value + 5)
    #     plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    #     # plt.title(f"Distribution of {license_type.capitalize()} License by Label")
    #     # plt.xlabel("Classification Label")
    #     plt.ylabel("Count")
    #     plt.tight_layout()
    #     plt.show()

    df = pd.DataFrame.from_dict(distribution, orient="index").fillna(0)
    df.plot(kind="bar", stacked=False, edgecolor="black", color=["#4c72b0" if lc == 'youtube' else "#dd8452" for lc in df.columns])
    plt.xticks(rotation=0)
    # plt.title("License Content Distribution by Label")
    plt.tight_layout()
    plt.show()

    return distribution

def stats_tag_scores(data):
    tag_values = [d.get("tagScores", []) for d in data]
    flattened_values = [score for sublist in tag_values for score in sublist]

    if not flattened_values:
        return 0, 0

    mean, var, std = np.mean(flattened_values), np.var(flattened_values), np.std(flattened_values)
    print(f"TagScores mean: {mean}, var: {var}, std: {std}")
    return mean, var, std

def show_tag_scores_distribution(raw_data, features_data):
    distribution = defaultdict(list)

    for raw_entry, feature_entry in zip(raw_data, features_data):
        label = raw_entry.get("classification_label", "Unknown")
        tag_scores = feature_entry.get("tagScores", [])
        if tag_scores:
            distribution[label].extend(tag_scores)

    label_order = ["suitable", "irrelevant", "restricted", "disturbing"]
    colors = {"suitable": "#0a70b0", "disturbing": "#f7951d", "restricted": "#9d1a1d", "irrelevant": "#115027"}
    # colors = {"suitable": "tab:blue", "disturbing": "tab:orange", "restricted": "tab:red", "irrelevant": "green"}

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.8), sharex=True, sharey=True)
    # 12, 8

    for ax, label in zip(axes.flat, label_order):
        values = distribution.get(label, [])
        if values:
            sns.kdeplot(values, ax=ax, fill=True, color=colors.get(label, "black"))
            ax.set_title(label.capitalize())
        ax.set_xlabel("Tag Score")
        ax.set_ylabel("Density")

    plt.tight_layout()
    plt.show()

    return distribution

    # for label, values in distribution.items():
    #     # plt.figure(figsize=(10, 5))
    #     sns.kdeplot(values, label=label, color=colors.get(label, "black"), fill=True)
    #     # plt.ylim(0, max_y_value)
    #     # plt.title(f"Tag Scores Distribution by Label - {label.capitalize()}")
    #     plt.xlabel("Tag Score")
    #     plt.ylabel("Density")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
    #
    # return distribution

def stats_thumbnail_emotions(data):
    emotions = ["anger", "fear", "joy", "sadness", "disgust", "trust", "anticipation", "positive", "negative", "surprise"]
    emotion_stats = {}

    for emotion in emotions:
        values = [float(d.get("thumbnailEmotions", {}).get(emotion, 0)) for d in data]
        emotion_stats[emotion] = {"mean": np.mean(values), "variance": np.var(values), "std": np.std(values)}
    print(f"Thumbnail emotions: {emotion_stats}")

    return emotion_stats

def stats_thumbnail_hard_words(data):
    values = [float(d.get("thumbnailHardWords", 0)) for d in data]
    mean, var, std = np.mean(values), np.var(values), np.std(values)
    print(f"Thumbnail hard words mean: {mean}, var: {var}, std: {std}")
    return mean, var

def map_category_id(category_id):
    category_map = {
        "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music", "15": "Pets & Animals",
        "17": "Sports", "18": "Short Movies", "19": "Travel & Events", "20": "Gaming",
        "21": "Videoblogging", "22": "People & Blogs", "23": "Comedy", "24": "Entertainment",
        "25": "News & Politics", "26": "Howto & Style", "27": "Education", "28": "Science & Technology",
        "29": "Nonprofits & Activism", "30": "Movies", "31": "Anime/Animation", "32": "Action/Adventure",
        "33": "Classics", "34": "Comedy", "35": "Documentary", "36": "Drama", "37": "Family",
        "38": "Foreign", "39": "Horror", "40": "Sci-Fi/Fantasy", "41": "Thriller", "42": "Shorts",
        "43": "Shows", "44": "Trailers"
    }
    return category_map.get(category_id, "Unknown")

############### Correlations ###################

def compute_cramers_v(chi2, n, k):
    return np.sqrt(chi2 / (n * k))

def compute_correlation_licensedcontent(data):
    distribution = defaultdict(lambda: defaultdict(int))

    for d in data:
        licensed = int(d.get("contentDetails", {}).get("licensedContent", 0))
        label = d.get("classification_label", "Unknown")
        distribution[label][licensed] += 1

    df = pd.DataFrame.from_dict(distribution, orient="index").fillna(0)

    chi2, p, _, _ = chi2_contingency(df)

    n = df.to_numpy().sum()
    k = min(df.shape) - 1
    cramers_v_value = compute_cramers_v(chi2, n, k)

    labels = []
    values = []
    unique_labels = sorted(df.index)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    for label, counts in distribution.items():
        for licensed, count in counts.items():
            labels.extend([label_mapping[label]] * count)
            values.extend([licensed] * count)

    mutual_info = mutual_info_score(labels, values)
    print(f"Chi-Square Test (Licensed Content): Chi² = {chi2:.4f}, p-value = {p:.4f}")
    print(f"Cramér’s V: {cramers_v_value:.4f}")
    print(f"Mutual Information Score: {mutual_info:.4f}\n")

    return {"feature": "licensedContent", "chi2": chi2, "p_value": p, "cramers_v": cramers_v_value, "mutual_info": mutual_info}

def compute_correlation_categoryid(data):
    distribution = defaultdict(lambda: defaultdict(int))

    for d in data:
        category = str(d.get("snippet", {}).get("categoryId", "Unknown"))
        label = d.get("classification_label", "Unknown")
        distribution[label][category] += 1

    df = pd.DataFrame.from_dict(distribution, orient="index").fillna(0)
    chi2, p, _, _ = chi2_contingency(df)

    n = df.to_numpy().sum()
    k = min(df.shape) - 1
    cramers_v_value = compute_cramers_v(chi2, n, k)

    labels = []
    values = []
    unique_labels = sorted(df.index)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    for label, counts in distribution.items():
        for category, count in counts.items():
            labels.extend([label_mapping[label]] * count)
            values.extend([category] * count)

    mutual_info = mutual_info_score(labels, values)

    print(f"Chi-Square Test (Category ID): Chi² = {chi2:.4f}, p-value = {p:.4f}")
    print(f"Cramér’s V: {cramers_v_value:.4f}")
    print(f"Mutual Information Score: {mutual_info:.4f}\n")

    return {"feature": "categoryId", "chi2": chi2, "p_value": p, "cramers_v": cramers_v_value, "mutual_info": mutual_info}

def compute_correlation_defaultaudiolanguage(data):
    distribution = defaultdict(lambda: defaultdict(int))

    for d in data:
        language = str(d.get("snippet", {}).get("defaultAudioLanguage", "Unknown"))
        label = d.get("classification_label", "Unknown")
        distribution[label][language] += 1

    df = pd.DataFrame.from_dict(distribution, orient="index").fillna(0)
    chi2, p, _, _ = chi2_contingency(df)

    n = df.to_numpy().sum()
    k = min(df.shape) - 1
    cramers_v_value = compute_cramers_v(chi2, n, k)

    labels = []
    values = []
    unique_labels = sorted(df.index)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    for label, counts in distribution.items():
        for language, count in counts.items():
            labels.extend([label_mapping[label]] * count)
            values.extend([language] * count)

    mutual_info = mutual_info_score(labels, values)

    print(f"Chi-Square Test (Default Audio Language): Chi² = {chi2:.4f}, p-value = {p:.4f}")
    print(f"Cramér’s V: {cramers_v_value:.4f}")
    print(f"Mutual Information Score: {mutual_info:.4f}\n")

    return {"feature": "defaultAudioLanguage", "chi2": chi2, "p_value": p, "cramers_v": cramers_v_value, "mutual_info": mutual_info}


def compute_correlation_topicchange(raw_data, features_data, feature_name):
    labels = []
    topic_change_values = []
    same_topic_values = []

    unique_labels = sorted(set(d["classification_label"] for d in raw_data))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    for raw_entry, feature_entry in zip(raw_data, features_data):
        label = raw_entry.get("classification_label", "Unknown")
        labels.append(label_mapping[label])

        topic_change_score = float(feature_entry.get(feature_name, {}).get("TOPICCHANGE", 0))
        same_topic_score = float(feature_entry.get(feature_name, {}).get("SAMETOPIC", 0))

        topic_change_values.append(topic_change_score)
        same_topic_values.append(same_topic_score)

    results = []

    spearman_corr_topic, spearman_p_topic = spearmanr(topic_change_values, labels)
    print(f"Spearman Correlation ({feature_name} - TOPICCHANGE): {spearman_corr_topic:.4f}, p-value: {spearman_p_topic:.4f}\n")
    results.append({"feature": f"{feature_name}.TOPICCHANGE", "spearman_corr": spearman_corr_topic, "p_value": spearman_p_topic})

    spearman_corr_same, spearman_p_same = spearmanr(same_topic_values, labels)
    print(f"Spearman Correlation ({feature_name} - SAMETOPIC): {spearman_corr_same:.4f}, p-value: {spearman_p_same:.4f}\n")
    results.append({"feature": f"{feature_name}.SAMETOPIC", "spearman_corr": spearman_corr_same, "p_value": spearman_p_same})

    return results

def compute_correlation_emotions(raw_data, features_data, feature_name):
    emotions = ["anger", "fear", "joy", "sadness", "disgust", "trust", "anticipation", "positive", "negative", "surprise"]
    labels = []
    emotion_values = {emotion: [] for emotion in emotions}

    unique_labels = sorted(set(d["classification_label"] for d in raw_data))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    for raw_entry, feature_entry in zip(raw_data, features_data):
        label = raw_entry.get("classification_label", "Unknown")
        labels.append(label_mapping[label])

        for emotion in emotions:
            emotion_score = float(feature_entry.get(feature_name, {}).get(emotion, 0))
            emotion_values[emotion].append(emotion_score)

    results = []
    for emotion in emotions:
        spearman_corr, spearman_p = spearmanr(emotion_values[emotion], labels)

        print(f"Spearman Correlation ({feature_name} - {emotion}): {spearman_corr:.4f}, p-value: {spearman_p:.4f}\n")

        results.append({"feature": f"{feature_name}.{emotion}", "spearman_corr": spearman_corr, "p_value": spearman_p})

    return results


def compute_correlation_emotions_combined(raw_data, features_data, feature_name, method="mean"):
    emotions = ["anger", "fear", "joy", "sadness", "disgust", "trust", "anticipation", "positive", "negative",
                "surprise"]
    labels = []
    combined_values = []

    unique_labels = sorted(set(d["classification_label"] for d in raw_data))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    for raw_entry, feature_entry in zip(raw_data, features_data):
        label = raw_entry.get("classification_label", "Unknown")
        labels.append(label_mapping[label])

        emotion_values = [float(feature_entry.get(feature_name, {}).get(emotion, 0)) for emotion in emotions]

        if method == "mean":
            combined_score = np.mean(emotion_values)
        elif method == "sum":
            combined_score = sum(emotion_values)
        elif method == "variance":
            combined_score = np.var(emotion_values)
        elif method == "max":
            combined_score = max(emotion_values)
        elif method == "polarity":
            combined_score = (
                                     feature_entry.get(feature_name, {}).get("positive", 0) +
                                     feature_entry.get(feature_name, {}).get("trust", 0) +
                                     feature_entry.get(feature_name, {}).get("joy", 0)
                             ) - (
                                     feature_entry.get(feature_name, {}).get("negative", 0) +
                                     feature_entry.get(feature_name, {}).get("fear", 0) +
                                     feature_entry.get(feature_name, {}).get("sadness", 0)
                             )
        else:
            raise ValueError("Invalid method. Choose 'mean', 'sum', 'variance', 'max', or 'polarity'.")

        combined_values.append(combined_score)

    spearman_corr, spearman_p = spearmanr(combined_values, labels)

    print(
        f"Spearman Correlation (Combined {feature_name}, method={method}): {spearman_corr:.4f}, p-value: {spearman_p:.4f}\n")

    return {"feature": feature_name, "spearman_corr": spearman_corr, "p_value": spearman_p}

def compute_correlation_hardwords(raw_data, features_data, feature_name):
    labels = []
    values = []

    unique_labels = sorted(set(d["classification_label"] for d in raw_data))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    for raw_entry, feature_entry in zip(raw_data, features_data):
        label = raw_entry.get("classification_label", "Unknown")
        hard_words_score = float(feature_entry.get(feature_name, 0))

        labels.append(label_mapping[label])
        values.append(hard_words_score)

    spearman_corr, spearman_p = spearmanr(values, labels)

    print(f"Spearman Correlation ({feature_name}): {spearman_corr:.4f}, p-value: {spearman_p:.4f}\n")

    return {"feature": feature_name, "spearman_corr": spearman_corr, "p_value": spearman_p}

def compute_correlation_statistics(feature_name, data):
    labels = []
    values = []

    unique_labels = sorted(set(d["classification_label"] for d in data))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    for d in data:
        value = int(d.get("statistics", {}).get(feature_name, 0))  # Extract numerical value
        label = d.get("classification_label", "Unknown")

        labels.append(label_mapping[label])
        values.append(value)

    spearman_corr, spearman_p = spearmanr(values, labels)
    mutual_info = mutual_info_score(labels, values)

    print(f"Spearman Correlation ({feature_name}): {spearman_corr:.4f}, p-value: {spearman_p:.4f}")
    print(f"Mutual Information Score: {mutual_info:.4f}\n")

    return {"feature": feature_name, "spearman_corr": spearman_corr, "p_value": spearman_p, "mutual_info": mutual_info}

def compute_correlation_license(data):
    distribution = defaultdict(lambda: defaultdict(int))

    for d in data:
        license_status = d.get("status", {}).get("license", "creativeCommon")
        distribution[d.get("classification_label")][license_status] += 1

    df = pd.DataFrame.from_dict(distribution, orient="index").fillna(0)
    chi2, p, _, _ = chi2_contingency(df)

    n = df.to_numpy().sum()
    k = min(df.shape) - 1
    cramers_v_value = compute_cramers_v(chi2, n, k)

    labels = []
    values = []
    unique_labels = sorted(df.index)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    for label, counts in distribution.items():
        for license_type, count in counts.items():
            labels.extend([label_mapping[label]] * count)
            values.extend([license_type] * count)

    mutual_info = mutual_info_score(labels, values)

    print(f"Chi-Square Test (License): Chi² = {chi2:.4f}, p-value = {p:.4f}")
    print(f"Cramér’s V: {cramers_v_value:.4f}")
    print(f"Mutual Information Score: {mutual_info:.4f}\n")

    return {"feature": "license", "chi2": chi2, "p_value": p, "cramers_v": cramers_v_value, "mutual_info": mutual_info}

def compute_correlation_tagscores(raw_data, features_data):
    labels = []
    values = []

    unique_labels = sorted(set(d["classification_label"] for d in raw_data))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    for raw_entry, feature_entry in zip(raw_data, features_data):
        label = raw_entry.get("classification_label", "Unknown")
        tag_scores = feature_entry.get("tagScores", [])

        avg_tag_score = np.mean(tag_scores) if tag_scores else 0

        labels.append(label_mapping[label])
        values.append(avg_tag_score)

    spearman_corr, spearman_p = spearmanr(values, labels)

    print(f"Spearman Correlation (Tag Scores): {spearman_corr:.4f}, p-value: {spearman_p:.4f}\n")

    return {"feature": "tagScores", "spearman_corr": spearman_corr, "p_value": spearman_p}

def analyze_all_features_one_result(raw_data, features_data):
    results = []

    results.append(compute_correlation_licensedcontent(raw_data))
    results.append(compute_correlation_categoryid(raw_data))
    results.append(compute_correlation_defaultaudiolanguage(raw_data))

    results.extend(compute_correlation_topicchange(raw_data, features_data, "titleTopicChange"))
    results.extend(compute_correlation_topicchange(raw_data, features_data, "descriptionTopicChange"))

    results.extend(compute_correlation_emotions(raw_data, features_data, "titleEmotions"))
    results.extend(compute_correlation_emotions(raw_data, features_data, "descriptionEmotions"))

    results.append(compute_correlation_hardwords(raw_data, features_data, "titleHardWords"))
    results.append(compute_correlation_hardwords(raw_data, features_data, "descriptionHardWords"))

    statistical_features = ["viewCount", "likeCount", "dislikeCount", "commentCount"]
    for feature in statistical_features:
        results.append(compute_correlation_statistics(feature, raw_data))

    results.append(compute_correlation_license(raw_data))
    results.append(compute_correlation_tagscores(raw_data, features_data))

    results.extend(compute_correlation_emotions(raw_data, features_data, "thumbnailEmotions"))
    results.append(compute_correlation_hardwords(raw_data, features_data, "thumbnailHardWords"))

    df_results = pd.DataFrame(results)

    # df_results["importance"] = df_results["cramers_v"].fillna(df_results["spearman_corr"])
    # df_results = df_results.sort_values(by="importance", ascending=False)
    #
    # print("\nRanked Feature Importance Based on Correlation:")
    # print(df_results[["feature", "importance", "mutual_info", "p_value"]])
    #
    # return df_results

    df_results["importance"] = df_results.apply(
        lambda row: abs(row["spearman_corr"]) if pd.notna(row["spearman_corr"]) else row["cramers_v"], axis=1
    )

    df_results = df_results.sort_values(by="importance", ascending=False)

    print("\nRanked Feature Importance Based on Correlation:")
    print(df_results[["feature", "importance", "mutual_info", "p_value"]])

    return df_results

def analyze_all_features(raw_data, features_data, emotion_method="mean"):
    categorical_results = []
    numerical_results = []

    categorical_results.append(compute_correlation_licensedcontent(raw_data))
    categorical_results.append(compute_correlation_license(raw_data))
    categorical_results.append(compute_correlation_categoryid(raw_data))
    categorical_results.append(compute_correlation_defaultaudiolanguage(raw_data))

    numerical_features = ["viewCount", "likeCount", "dislikeCount", "commentCount"]
    for feature in numerical_features:
        numerical_results.append(compute_correlation_statistics(feature, raw_data))

    numerical_results.extend(compute_correlation_topicchange(raw_data, features_data, "titleTopicChange"))
    numerical_results.extend(compute_correlation_topicchange(raw_data, features_data, "descriptionTopicChange"))

    # numerical_results.append(compute_correlation_emotions_combined(raw_data, features_data, "titleEmotions", emotion_method))
    # numerical_results.append(compute_correlation_emotions_combined(raw_data, features_data, "descriptionEmotions", emotion_method))
    numerical_results.extend(compute_correlation_emotions(raw_data, features_data, "titleEmotions"))
    numerical_results.extend(compute_correlation_emotions(raw_data, features_data, "descriptionEmotions"))

    numerical_results.append(compute_correlation_hardwords(raw_data, features_data, "titleHardWords"))
    numerical_results.append(compute_correlation_hardwords(raw_data, features_data, "descriptionHardWords"))

    numerical_results.append(compute_correlation_tagscores(raw_data, features_data))

    # numerical_results.append(compute_correlation_emotions_combined(raw_data, features_data, "thumbnailEmotions", emotion_method))
    numerical_results.extend(compute_correlation_emotions(raw_data, features_data, "thumbnailEmotions"))
    numerical_results.append(compute_correlation_hardwords(raw_data, features_data, "thumbnailHardWords"))

    all_categorical = [item for sublist in categorical_results for item in (sublist if isinstance(sublist, list) else [sublist])]
    all_numerical = [item for sublist in numerical_results for item in (sublist if isinstance(sublist, list) else [sublist])]

    df_categorical = pd.DataFrame(all_categorical)
    df_numerical = pd.DataFrame(all_numerical)

    df_categorical = df_categorical.dropna(subset=["feature"])
    df_numerical = df_numerical.dropna(subset=["feature"])

    df_categorical = df_categorical.sort_values(by="cramers_v", ascending=False)

    df_numerical["importance"] = df_numerical["spearman_corr"].abs()
    df_numerical = df_numerical.sort_values(by="importance", ascending=False)

    # print("\nRanked Categorical Feature Importance (Cramér’s V):")
    # print(df_categorical[["feature", "cramers_v", "p_value"]])
    #
    # print("\nRanked Numerical Feature Importance (Spearman’s ρ):")
    # print(df_numerical[["feature", "spearman_corr", "importance", "p_value"]])

    df_combined = pd.concat([df_categorical, df_numerical], ignore_index=True)

    df_combined["importance"] = df_combined["importance"].fillna(df_combined["cramers_v"])
    df_combined = df_combined.sort_values(by="importance", ascending=False)

    return df_combined

def select_features_by_correlation(df_results, threshold=0.1, p_value_cutoff=0.05):
    df_filtered = df_results[
        (df_results["importance"] >= threshold) & (df_results["p_value"] < p_value_cutoff)
    ]

    print("\nSelected Features Based on Correlation Threshold:")
    print(df_filtered[["feature", "importance", "p_value"]])

    return df_filtered

if __name__ == "__main__":
    raw_data = load_dataset("../datasets/remaining_data.json")
    features_data = load_dataset("../datasets/remaining_data_features.json")

    stats_licensedcontent(raw_data)
    show_licensedcontent_distribution(raw_data)

    stats_categoryid(raw_data)
    show_categoryid_distribution(raw_data)
    show_categoryid_distribution_stacked_numbers(raw_data)
    show_categoryid_distribution_stacked_percentage(raw_data)

    stats_default_audio_language(raw_data)
    show_default_audio_language_distribution(raw_data)
    show_default_audio_language_distribution_stacked_numbers(raw_data)
    show_default_audio_language_distribution_stacked_percentage(raw_data)

    stats_title_topicchange(features_data)
    show_topicchange_distribution(raw_data, features_data, "titleTopicChange", "Title Topic Change Distribution")

    stats_description_topicchange(features_data)
    show_topicchange_distribution(raw_data, features_data, "descriptionTopicChange", "Description Topic Change Distribution")

    stats_title_emotions(features_data)
    show_emotions_distribution(raw_data, features_data, "titleEmotions", "Title Emotions Distribution", True)
    show_emotions_distribution(raw_data, features_data, "titleEmotions", "Title Emotions Distribution", False)

    stats_description_emotions(features_data)
    show_emotions_distribution(raw_data, features_data, "descriptionEmotions", "Description Emotions Distribution", True)
    show_emotions_distribution(raw_data, features_data, "descriptionEmotions", "Description Emotions Distribution", False)

    stats_title_hard_words(features_data)
    show_hardwords_distribution(raw_data, features_data, "titleHardWords", "hard words title")
    show_hardwords_distribution_stacked(raw_data, features_data, "titleHardWords", "hard words title")

    stats_description_hard_words(features_data)
    show_hardwords_distribution(raw_data, features_data, "descriptionHardWords", "hard words description")
    show_hardwords_distribution_stacked(raw_data, features_data, "descriptionHardWords", "hard words description")

    stats_statistics(raw_data, "viewCount")
    show_statistics_distribution(raw_data, "viewCount", "views", "blue", 150)
    show_statistics_distribution_cdf(raw_data, "viewCount", "views")

    stats_statistics(raw_data, "likeCount")
    show_statistics_distribution(raw_data, "likeCount", "likes", "red", 100)
    show_statistics_distribution_cdf(raw_data, "likeCount", "likes")

    stats_statistics(raw_data, "dislikeCount")
    show_statistics_distribution(raw_data, "dislikeCount", "dislikes", "green", 15)
    show_statistics_distribution_cdf(raw_data, "dislikeCount", "dislikes")

    stats_statistics(raw_data, "commentCount")
    show_statistics_distribution(raw_data, "commentCount", "comments", "yellow", 30)
    show_statistics_distribution_cdf(raw_data, "commentCount", "comments")

    stats_statistics(raw_data, "favoriteCount")

    stats_license(raw_data)
    show_license_distribution(raw_data)

    stats_tag_scores(features_data)
    show_tag_scores_distribution(raw_data, features_data)

    stats_thumbnail_emotions(features_data)
    show_emotions_distribution(raw_data, features_data, "thumbnailEmotions", "Thumbnail Emotions Distribution", True)
    show_emotions_distribution(raw_data, features_data, "thumbnailEmotions", "Thumbnail Emotions Distribution", False)

    stats_thumbnail_hard_words(features_data)
    show_hardwords_distribution(raw_data, features_data, "thumbnailHardWords", "hard words thumbnail")
    show_hardwords_distribution_stacked(raw_data, features_data, "thumbnailHardWords", "hard words thumbnail")

    # analyze_all_features_one_result(raw_data, features_data)
    df_correlations = analyze_all_features(raw_data, features_data, "mean")
    # print(df_correlations[["feature", "importance", "p_value"]])

    select_features_by_correlation(df_correlations, threshold=0.1, p_value_cutoff=0.05)