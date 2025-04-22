import json
from collections import defaultdict

from matplotlib import pyplot as plt

def load_dataset_original(filepath):
    with open(filepath, "r+", encoding="UTF-8") as file:
        data = [json.loads(line) for line in file.readlines()]
    return data

def load_dataset(filepath):
    with open(filepath, "r+", encoding="UTF-8") as file:
        data = json.loads(file.read())
    return data

def save_to_file(data, filepath):
    with open(filepath, "w+", encoding="UTF-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def find_videos_in_dataset(unknown_list, complete_dataset, groundtruth_list):
    found_videos = {}

    for video in complete_dataset:
        video_id = video.get("id")
        if video_id in unknown_list and video_id not in groundtruth_list:
            found_videos[video_id] = video

    return found_videos

def find_distribution_unknown_related_videos(list):
    # distribution = {}
    distribution = defaultdict(int)
    distributionwo = defaultdict(int)
    non_unknown_related_videos = []
    count = 0

    for video_id, related_videos in list.items():
        unknown_count = 0
        none_count = 0
        lengthrelvid = 0

        for related_video in related_videos:
            groundtruth_label = related_video["groundtruth_label"]
            if groundtruth_label == "unknown":
                unknown_count += 1
            elif groundtruth_label is None:
                none_count += 1
                print("Found None Value")
            else:
                none_count += 1
                lengthrelvid += 1

        distribution[len(related_videos)] += 1
        distributionwo[lengthrelvid] += 1

        # distribution[unknown_count] += 1
        # distribution[video_id] = {
        #     "unknown_count": unknown_count,
        # }

        # if unknown_count == 0:
        #     non_unknown_related_videos.append(video_id)

        # lengthrelvid += len(related_videos)
        # count += 1

    plt.bar(distribution.keys(), distribution.values(), edgecolor="black", color="#4c72b0")
    plt.xlabel("Length of list")
    plt.ylabel("# of lists")
    plt.tight_layout()
    plt.show()

    # plt.bar(distributionwo.keys(), distributionwo.values(), edgecolor="black", color="#4c72b0")
    # plt.xlabel("Length of list without unknowns")
    # plt.ylabel("# of lists")
    # plt.tight_layout()
    # plt.show()

    # print(f"\nnumber of related videos without unknowns: {lengthrelvid}")
    # print(f"\nnumber of lists: {count}")
    # print(f"\naverage length of lists: {lengthrelvid/count}")

    # save_to_file(distribution, "unknown_videos_map.json")

    return distribution, non_unknown_related_videos


if __name__ == "__main__":
    path_list = "ranked_videos.json"
    list = load_dataset(path_list)
    #
    # unknown_list = []
    #
    # for vid_id, rel_list in list.items():
    #     for item in rel_list:
    #         if item.get("groundtruth_label") == "unknown":
    #             unknown_list.append(item.get("video_id"))
    #
    # unknown_list = [set(unknown_list)]
    #
    # elsagate = load_dataset_original("../disturbed_youtube_dataset/elsagate_related_videos.json")
    # other = load_dataset_original("../disturbed_youtube_dataset/other_child_related_videos.json")
    # popular = load_dataset_original("../disturbed_youtube_dataset/popular_videos.json")
    # random = load_dataset_original("../disturbed_youtube_dataset/random_videos.json")
    # complete_dataset = elsagate + other + popular + random
    #
    # ground_truth = load_dataset_original("../disturbed_youtube_dataset/groundtruth_videos.json")
    # groundtruth_map = [d.get("id") for d in ground_truth]
    #
    # result = find_videos_in_dataset(unknown_list, complete_dataset, ground_truth)
    #
    # save_to_file(result, "../data/datasets/related_videos_unknowns.json")
    #
    # print(f"\nFound {len(result)} unknown videos in the complete dataset.")

    distribution, non_unknown_related_videos = find_distribution_unknown_related_videos(list)
