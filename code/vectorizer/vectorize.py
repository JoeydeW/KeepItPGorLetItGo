import json
from nrclex import NRCLex
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

from tqdm import tqdm
import os

from classifier import data_processing
from hard_words.calc_hard_words import get_hard_words

hugging_face_token = "INPUT YOUR TOKEN HERE"

def load_dataset(filepath):
    with open(filepath, "r+", encoding="UTF-8") as file:
        data = json.loads(file.read())
    return data

def save_to_file(data, filepath):
    with open(filepath, "w+", encoding="UTF-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def prep_feature(data, feature):
    features = set()
    for dataset in data:
        for sample in dataset:
            for f in feature:
                sample = sample.get(f)
            features.add(sample)
    return list(features)

def prep_feature_dataset(data, feature):
    features = set()
    for sample in data:
        for f in feature:
            sample = sample.get(f)
        features.add(sample)
    return list(features)

def one_hot_encode(value, values):
    vector = [0] * len(values)
    if value in values:
        vector[values.index(value)] = 1
    return vector

def split_text_into_sequences(text, max_tokens=512):
  return [text[i:i + max_tokens] for i in range(0, len(text), max_tokens)]

def process_text_with_pipeline(pipe, text):
  sequences = split_text_into_sequences(text.strip(), max_tokens=512)
  if not sequences:
    return {"SAMETOPIC": 0.0, "TOPICCHANGE": 0.0}

  scores = {"SAMETOPIC": 0.0, "TOPICCHANGE": 0.0}
  for seq in sequences:
    result = pipe(seq)[0]
    scores[result["label"]] += result["score"]

  total_sequences = len(sequences)
  if total_sequences > 0:
    for label in scores:
      scores[label] /= total_sequences

  return scores

def text_processing(text):
    pipe = pipeline("text-classification", model="raicrits/topicChangeDetector_v1", token=hugging_face_token)
    topic_change = process_text_with_pipeline(pipe, text)
    return [topic_change.get("SAMETOPIC"), topic_change.get("TOPICCHANGE")]

def emotion_processing(text):
    text_emotions = NRCLex(text).affect_frequencies
    return [text_emotions.get("anger"), text_emotions.get("fear"),
            text_emotions.get("joy"), text_emotions.get("sadness"),
            text_emotions.get("disgust"), text_emotions.get("trust"),
            text_emotions.get("anticip"), text_emotions.get("positive"),
            text_emotions.get("negative"), text_emotions.get("surprise")]

def hard_words_processing(text):
    return get_hard_words(text)[2]

def prep_tags(data):
    tag_strings = [" ".join(d.get("snippet", {}).get("tags", {})) for dataset in data for d in dataset]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tag_strings)

    feature_names = vectorizer.get_feature_names_out()
    tag_importance = tfidf_matrix.sum(axis=0).A1

    tag_scores = dict(zip(feature_names, tag_importance))
    return tag_scores

def prep_tags_dataset(data):
    tag_strings = [" ".join(d.get("snippet", {}).get("tags", {})) for d in data]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tag_strings)

    feature_names = vectorizer.get_feature_names_out()
    tag_importance = tfidf_matrix.sum(axis=0).A1

    tag_scores = dict(zip(feature_names, tag_importance))
    return tag_scores

def process_tags(tags, tag_scores, n=5):
    tags_with_scores = [(tag, tag_scores.get(tag, 0)) for tag in tags]
    top_n_tags = list(sorted(tags_with_scores, key=lambda x: x[1], reverse=True))[:n]

    nlp = spacy.load("en_core_web_md")
    top_n_tags = [nlp(tag).vector_norm for tag, _ in top_n_tags]

    while len(top_n_tags) < n:
        top_n_tags.append(0)

    return top_n_tags

def process_thumbnail(url):
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", token=hugging_face_token)
    caption = captioner(url)[0]["generated_text"]
    emotion = emotion_processing(caption)
    hard_words = hard_words_processing(caption)
    emotion.append(hard_words)
    return emotion

def get_thumbnail(thumbnail):
    if thumbnail.get("maxres") is not None:
        return  thumbnail.get("maxres").get("url")
    elif thumbnail.get("standard") is not None:
        return thumbnail.get("standard").get("url")
    elif thumbnail.get("high") is not None:
        return thumbnail.get("high").get("url")
    elif thumbnail.get("medium") is not None:
        return thumbnail.get("medium").get("url")
    elif thumbnail.get("default") is not None:
        return thumbnail.get("default").get("url")
    else:
        return ""


def vectorize_fold(data, categories, languages, tag_scores):
    result = []
    failed = []

    for e, fold in enumerate(data):
        vectorized_fold = []
        for r, dataset in enumerate(fold):
            vectors = []
            for d in tqdm(dataset, desc=f"Fold {e + 1}, Dataset {r + 1}", total=len(dataset)):
                try:
                    vector = []
                    vector.append(int(d.get("contentDetails").get("licensedContent")))
                    vector.extend(one_hot_encode(d.get("snippet").get("categoryId"), categories))
                    vector.extend(one_hot_encode(d.get("snippet").get("defaultAudioLanguage"), languages))
                    vector.extend(text_processing(d.get("snippet").get("title")))
                    vector.extend(text_processing(d.get("snippet").get("description")))
                    vector.extend(emotion_processing(d.get("snippet").get("title")))
                    vector.extend(emotion_processing(d.get("snippet").get("description")))
                    vector.append(hard_words_processing(d.get("snippet").get("title")))
                    vector.append(hard_words_processing(d.get("snippet").get("description")))
                    vector.append(d.get("statistics", {}).get("viewCount", 0))
                    vector.append(d.get("statistics", {}).get("likeCount", 0))
                    vector.append(d.get("statistics", {}).get("dislikeCount", 0))
                    vector.append(d.get("statistics", {}).get("favoriteCount", 0))
                    vector.append(d.get("statistics", {}).get("commentCount", 0))
                    vector.append(int(d.get("status", {}).get("license") == "creativeCommon"))
                    vector.extend(process_tags(d.get("snippet").get("tags", {}), tag_scores))
                    vector.extend(process_thumbnail(d.get("snippet").get("thumbnails", {}).get("standard", {}).get("url")))
                    vectors.append(vector)
                except:
                    failed.append(d)
            vectorized_fold.append(vectors)
        result.append(vectorized_fold)

    return result, failed

def vectorize_data(data, categories, languages, tag_scores):
    result = []
    failed = []

    for i, d in enumerate(tqdm(data, desc=f"Processing Data", unit="entry", total=len(data))):
        try:
            vector = []
            vector.append(int(d.get("contentDetails", {}).get("licensedContent", False)))
            vector.extend(one_hot_encode(d.get("snippet", {}).get("categoryId"), categories))
            vector.extend(one_hot_encode(d.get("snippet", {}).get("defaultAudioLanguage"), languages))
            vector.extend(text_processing(d.get("snippet", {}).get("title", "")))
            vector.extend(text_processing(d.get("snippet", {}).get("description", "")))
            vector.extend(emotion_processing(d.get("snippet", {}).get("title", "")))
            vector.extend(emotion_processing(d.get("snippet", {}).get("description", "")))
            vector.append(hard_words_processing(d.get("snippet", {}).get("title", "")))
            vector.append(hard_words_processing(d.get("snippet", {}).get("description", "")))
            vector.append(d.get("statistics", {}).get("viewCount", 0))
            vector.append(d.get("statistics", {}).get("likeCount", 0))
            vector.append(d.get("statistics", {}).get("dislikeCount", 0))
            vector.append(d.get("statistics", {}).get("favoriteCount", 0))
            vector.append(d.get("statistics", {}).get("commentCount", 0))
            vector.append(int(d.get("status", {}).get("license") == "creativeCommon"))
            vector.extend(process_tags(d.get("snippet", {}).get("tags", {}), tag_scores))
            vector.extend(process_thumbnail(get_thumbnail(d.get("snippet", {}).get("thumbnails", {}))))
            if(len(vector) != 114):
                print(
                    f"Error processing entry {i}: vector length is {len(vector)}, expected 114"
                )
                raise Exception("Vector length mismatch")
            result.append(vector)
        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            failed.append(d)

    return result, failed

def process_in_chunks(data, chunk_size, categories, languages, tag_scores, output_dir="output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_chunks = (len(data) + chunk_size - 1) // chunk_size

    completed_chunks = {
        int(f.split("_")[-1].split(".")[0])
        for f in os.listdir(output_dir)
        if f.startswith("vectorized_data_chunk_")
    }

    for chunk_idx in range(total_chunks):
        # Skip if the chunk is already processed
        if chunk_idx + 1 in completed_chunks:
            print(f"Skipping already processed chunk {chunk_idx + 1}/{total_chunks}")
            continue

        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(data))
        chunk = data[start_idx:end_idx]

        print(f"Processing chunk {chunk_idx + 1}/{total_chunks} (entries {start_idx}-{end_idx - 1})...")

        try:
            result, failed = vectorize_data(chunk, categories, languages, tag_scores)

            save_to_file(result, f"{output_dir}/vectorized_data_chunk_{chunk_idx + 1}.json")
            save_to_file(failed, f"{output_dir}/failed_data_chunk_{chunk_idx + 1}.json")

            print(f"Chunk {chunk_idx + 1}/{total_chunks} processed and saved.")
        except Exception as e:
            print(f"Error processing chunk {chunk_idx + 1}: {e}")
            break

def combine_chunks(input_dir, output_file):
    combined_result = []

    chunk_files = sorted([f for f in os.listdir(input_dir) if f.startswith("vectorized_data_chunk_")])

    for chunk_file in chunk_files:
        with open(os.path.join(input_dir, chunk_file), "r+", encoding="UTF-8") as file:
            chunk_data = json.load(file)
            combined_result.extend(chunk_data)

    save_to_file(combined_result, output_file)

    print(f"Combined data saved to {output_file}")
    print(f"Total entries: {len(combined_result)}")
    return combined_result

def vectorize(data):
    categories = prep_feature_dataset(data, ["snippet", "categoryId"])
    languages = prep_feature_dataset(data, ["snippet", "defaultAudioLanguage"])
    tag_scores = prep_tags_dataset(data)

    chunk_size = 100
    process_in_chunks(data, chunk_size, categories, languages, tag_scores, output_dir="vectorized_chunks")

    input_dir = "vectorized_chunks"
    output_file = "../old/vectorized_data_old.json"

    combined_result = combine_chunks(input_dir, output_file)

    return combined_result

def save_feature_values(data):
    tag_scores = prep_tags_dataset(data)
    pipe = pipeline("text-classification", model="raicrits/topicChangeDetector_v1", token=hugging_face_token)
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", token=hugging_face_token)

    result = []
    failed = []

    for i, d in enumerate(tqdm(data, desc="Processing Data", unit="entry", total=len(data))):
        try:
            caption = captioner(get_thumbnail(d.get("snippet", {}).get("thumbnails", {})))[0]["generated_text"]
            vector = {
                "titleTopicChange": process_text_with_pipeline(pipe, d.get("snippet", {}).get("title", "")),
                "descriptionTopicChange": process_text_with_pipeline(pipe, d.get("snippet", {}).get("description", "")),
                "titleEmotions": NRCLex(d.get("snippet", {}).get("title", "")).affect_frequencies,
                "descriptionEmotions": NRCLex(d.get("snippet", {}).get("description", "")).affect_frequencies,
                "titleHardWords": hard_words_processing(d.get("snippet", {}).get("title", "")),
                "descriptionHardWords": hard_words_processing(d.get("snippet", {}).get("description", "")),
                "tagScores": process_tags(d.get("snippet", {}).get("tags", {}), tag_scores),
                "thumbnailEmotions": NRCLex(caption).affect_frequencies,
                "thumbnailHardWords": hard_words_processing(caption),
            }
            result.append(vector)
        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            failed.append(d)

    return result, failed

def save_feature_values_in_chunks(data, chunk_size=100, output_filepath="feature_values.json"):
    tag_scores = prep_tags_dataset(data)

    pipe = pipeline("text-classification", model="raicrits/topicChangeDetector_v1", token=hugging_face_token)
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", token=hugging_face_token)

    if os.path.exists(output_filepath):
        with open(output_filepath, "r", encoding="utf-8") as file:
            processed_data = json.load(file)
    else:
        processed_data = []

    total_chunks = (len(data) + chunk_size - 1) // chunk_size

    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(data))
        chunk = data[start_idx:end_idx]

        print(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({start_idx}-{end_idx})...")

        result = []
        failed = []

        for i, d in enumerate(tqdm(chunk, desc=f"Chunk {chunk_idx + 1}", unit="entry", total=len(chunk))):
            try:
                caption = captioner(get_thumbnail(d.get("snippet", {}).get("thumbnails", {})))[0]["generated_text"]
                vector = {
                    "titleTopicChange": process_text_with_pipeline(pipe, d.get("snippet", {}).get("title", "")),
                    "descriptionTopicChange": process_text_with_pipeline(pipe, d.get("snippet", {}).get("description", "")),
                    "titleEmotions": NRCLex(d.get("snippet", {}).get("title", "")).affect_frequencies,
                    "descriptionEmotions": NRCLex(d.get("snippet", {}).get("description", "")).affect_frequencies,
                    "titleHardWords": hard_words_processing(d.get("snippet", {}).get("title", "")),
                    "descriptionHardWords": hard_words_processing(d.get("snippet", {}).get("description", "")),
                    "tagScores": process_tags(d.get("snippet", {}).get("tags", {}), tag_scores),
                    "thumbnailEmotions": NRCLex(caption).affect_frequencies,
                    "thumbnailHardWords": hard_words_processing(caption),
                }
                result.append(vector)
            except Exception as e:
                print(f"Error processing entry {start_idx + i}: {e}")
                failed.append(d)

        processed_data.extend(result)

        with open(output_filepath, "w", encoding="utf-8") as file:
            json.dump(processed_data, file, indent=2, ensure_ascii=False)

    print("Feature extraction completed and saved successfully!")

    return processed_data, failed


if __name__ == "__main__":
    dataset = "remaining_data_features.json"
    data = load_dataset(dataset)
    vectorized_data = vectorize(data)

    result, failed = save_feature_values_in_chunks(data)
    save_to_file(result, "remaining_data_features.json")

    stratified_folds = data_processing.stratified_kfold_split(vectorized_data, data, 8)
    save_to_file(stratified_folds, "stratified_folds_all_features.json")

    # dataset = "../data/datasets/related_videos_set.json"
    # data = load_dataset(dataset)
    #
    # result, failed = save_feature_values_in_chunks(data)
    # save_to_file(result, "related_videos_features.json")
