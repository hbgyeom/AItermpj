# 직접 녹음 데이터를 6초 간격으로 나누어 저장
import subprocess
import os


def split_wav(file_path, output_dir, interval=6):
    try:
        os.makedirs(output_dir, exist_ok=True)

        subprocess.run(
            ['ffmpeg', '-i', file_path, '-f', 'segment',
             '-segment_time', str(interval), '-c', 'copy',
             os.path.join(output_dir, 'part_%03d.wav')],
            check=True
        )
        print("Saved")
    except subprocess.CalledProcessError as e:
        print(f"Error during splitting: {e}")


wav_file = "C:/Users/HBG/codes/project/detection/combined.wav"
output_directory = "splits"
split_wav(wav_file, output_directory, interval=6)
