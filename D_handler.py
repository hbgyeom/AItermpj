import librosa
import numpy as np
import pandas as pd


def load_audio(file_path, csv_name):
    df = pd.read_csv(file_path + csv_name, encoding="utf-8")
    return df["filename"].tolist()


def crop_audio(audio_list, file_path, threshold):
    crop_list = []
    for i, audio_name in enumerate(audio_list, start=1):
        y, sr = librosa.load(file_path + audio_name,
                             sr=16000, duration=threshold)
        audio_length = len(y) / sr
        if audio_length < threshold:
            continue
        y_cropped = y[:sr * threshold]
        crop_list.append(y_cropped)
        print(f"\rCropping audio file {i}/{len(audio_list)}",
              end="", flush=True)
    print()
    return crop_list


def to_mfcc(crop_list):
    mfcc_list = []
    for i, audio_numpy in enumerate(crop_list, start=1):
        mfcc = librosa.feature.mfcc(y=audio_numpy, sr=16000, n_mfcc=13)
        mfcc_list.append(mfcc)
        print(f"\rProcessing audio file {i}/{len(crop_list)}",
              end="", flush=True)
    print()
    return mfcc_list


def save_mfcc(mfcc_list, file_path):
    mfcc_array = np.array(mfcc_list, dtype=object)
    np.save(file_path + "mfcc.npy", mfcc_array)
    print("File saved")


if __name__ == "__main__":
    file_path = "C:/Users/HBG/codes/project/dataset/D/data/"
    csv_name = "audio_list.csv"
    audio_list = load_audio(file_path, csv_name)
    crop_list = crop_audio(audio_list, file_path, threshold=5)
    mfcc_list = to_mfcc(crop_list)
    save_mfcc(mfcc_list, file_path)
