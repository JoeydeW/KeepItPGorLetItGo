import json
from tqdm import tqdm
import vectorize

def load_dataset(filepath):
    with open(filepath, "r", encoding="UTF-8") as file:
        return json.load(file)

def save_to_file(data, filepath):
    with open(filepath, "w+", encoding="UTF-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def vectorize_data_12_features(raw_data, features_data, categories, languages):
    result = []
    failed = []

    for i, (raw_entry, feature_entry) in enumerate(tqdm(zip(raw_data, features_data), desc="Processing Data", unit="entry", total=len(raw_data))):
        try:
            vector = []
            vector.extend(vectorize.one_hot_encode(raw_entry.get("snippet", {}).get("categoryId"), categories))
            vector.append(raw_entry.get("statistics", {}).get("viewCount", 0))
            vector.append(int(raw_entry.get("contentDetails", {}).get("licensedContent", False)))
            vector.extend(vectorize.one_hot_encode(raw_entry.get("snippet", {}).get("defaultAudioLanguage"), languages))
            vector.append(feature_entry.get("descriptionEmotions", {}).get("joy", 0.0))
            vector.append(raw_entry.get("statistics", {}).get("dislikeCount", 0))
            vector.append(feature_entry.get("titleEmotions", {}).get("disgust", 0.0))
            vector.append(feature_entry.get("titleEmotions", {}).get("negative", 0.0))
            vector.append(raw_entry.get("statistics", {}).get("likeCount", 0))
            vector.append(feature_entry.get("descriptionEmotions", {}).get("positive", 0.0))
            vector.append(feature_entry.get("descriptionEmotions", {}).get("trust", 0.0))
            vector.append(feature_entry.get("descriptionEmotions", {}).get("anticipation", 0.0))

            expected_length = 75
            if len(vector) != expected_length:
                print(f"Error processing entry {i}: vector length is {len(vector)}, expected {expected_length}")
                raise Exception("Vector length mismatch")

            result.append({raw_entry.get("id", ""): vector})

        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            failed.append(raw_entry)

    return result, failed

def vectorize_data_8_features(raw_data, features_data):
    result = []
    failed = []

    for i, (raw_entry, feature_entry) in enumerate(tqdm(zip(raw_data, features_data), desc="Processing Data", unit="entry", total=len(raw_data))):
        try:
            vector = []
            vector.extend(vectorize.one_hot_encode(raw_entry.get("snippet", {}).get("categoryId"), categories))
            vector.append(raw_entry.get("statistics", {}).get("viewCount", 0))
            vector.append(int(raw_entry.get("contentDetails", {}).get("licensedContent", False)))
            vector.extend(vectorize.one_hot_encode(raw_entry.get("snippet", {}).get("defaultAudioLanguage"), languages))
            vector.append(feature_entry.get("descriptionEmotions", {}).get("joy", 0.0))
            vector.append(raw_entry.get("statistics", {}).get("dislikeCount", 0))
            vector.append(feature_entry.get("titleEmotions", {}).get("disgust", 0.0))
            vector.append(feature_entry.get("titleEmotions", {}).get("negative", 0.0))

            expected_length = 71
            if len(vector) != expected_length:
                print(f"Error processing entry {i}: vector length is {len(vector)}, expected {expected_length}")
                raise Exception("Vector length mismatch")

            result.append(vector)

        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            failed.append(raw_entry)

    return result, failed

def vectorize_data_5_features(raw_data, features_data):
    result = []
    failed = []

    for i, (raw_entry, feature_entry) in enumerate(tqdm(zip(raw_data, features_data), desc="Processing Data", unit="entry", total=len(raw_data))):
        try:
            vector = []
            vector.extend(vectorize.one_hot_encode(raw_entry.get("snippet", {}).get("categoryId"), categories))
            vector.append(raw_entry.get("statistics", {}).get("viewCount", 0))
            vector.append(int(raw_entry.get("contentDetails", {}).get("licensedContent", False)))
            vector.extend(vectorize.one_hot_encode(raw_entry.get("snippet", {}).get("defaultAudioLanguage"), languages))
            vector.append(feature_entry.get("descriptionEmotions", {}).get("joy", 0.0))

            expected_length = 68
            if len(vector) != expected_length:
                print(f"Error processing entry {i}: vector length is {len(vector)}, expected {expected_length}")
                raise Exception("Vector length mismatch")

            result.append(vector)

        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            failed.append(raw_entry)

    return result, failed

def vectorize_data_4_features(raw_data, features_data):
    result = []
    failed = []

    for i, (raw_entry, feature_entry) in enumerate(tqdm(zip(raw_data, features_data), desc="Processing Data", unit="entry", total=len(raw_data))):
        try:
            vector = []
            vector.extend(vectorize.one_hot_encode(raw_entry.get("snippet", {}).get("categoryId"), categories))
            vector.append(raw_entry.get("statistics", {}).get("viewCount", 0))
            vector.append(int(raw_entry.get("contentDetails", {}).get("licensedContent", False)))
            vector.extend(vectorize.one_hot_encode(raw_entry.get("snippet", {}).get("defaultAudioLanguage"), languages))

            expected_length = 67
            if len(vector) != expected_length:
                print(f"Error processing entry {i}: vector length is {len(vector)}, expected {expected_length}")
                raise Exception("Vector length mismatch")

            result.append(vector)

        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            failed.append(raw_entry)

    return result, failed

if __name__ == "__main__":
    raw_data = load_dataset("../data/datasets/remaining_data.json")
    features_data = load_dataset("../data/datasets/remaining_data_features.json")

    categories = vectorize.prep_feature_dataset(raw_data, ["snippet", "categoryId"])
    languages = vectorize.prep_feature_dataset(raw_data, ["snippet", "defaultAudioLanguage"])

    # data = load_dataset("../data/datasets/related_videos_set.json")
    # features = load_dataset("../vectorizer/related_videos_features.json")

    vectorized_data, failed_data = vectorize_data_12_features(raw_data, features_data, categories, languages)

    save_to_file(vectorized_data, "../data/vectorized_data/related_videos_set_12_features.json")