from datahandler import get_json, get_wav

path = "C:/Users/HBG/codes/project/dataset/D/label/"
name = "label_list.csv"
json_list = get_json(path, name, 940)
wav_list = get_wav(path, name, 940)

print(json_list)
print(wav_list)
