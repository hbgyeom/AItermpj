import pandas as pd
import json


def get_playTimes(file_path, file_name):
    """
    "C:/Users/HBG/codes/project/dataset/D/label/"
    "label_list.csv"
    """
    label_list = pd.read_csv(file_path + file_name, encoding="utf-8")
    playTimes = []
    for filename in label_list["filename"]:
        try:
            with open(file_path + filename, 'r', encoding="utf-8") as file:
                json_data = json.load(file)
                playTime = json_data.get("playTime", None)
                if playTime is not None:
                    playTimes.append((filename, playTime))
        except Exception as e:
            print(e)
            continue
    return playTimes


def filter_playTime(playtime_list, threshold):
    return [filename for filename,
            playtime in playtime_list if playtime < threshold]


def json_to_wav(filter_list):
    wav_list = [filename.replace(".json", ".wav") for filename in filter_list]
    return wav_list


def get_wav(file_path, file_name, threshold):
    playTime_list = get_playTimes(file_path, file_name)
    filter_list = filter_playTime(playTime_list, 940)
    wav_list = json_to_wav(filter_list)
    return wav_list
