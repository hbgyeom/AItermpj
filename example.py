from datahandler import get_wav

path = "아래 csv파일이 있는 경로/"
name = "label_list.csv"
wav_list = get_wav(path, name, 940)

print(wav_list)
