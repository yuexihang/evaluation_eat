from glob import glob
import os
from tqdm import tqdm

allmp4s = glob('/data3/lipread_test_25/video/*.mp4')
path_wav='./talking_head_testing/lipread_test_wav/'
os.makedirs(path_wav, exist_ok=True)

for mp4 in tqdm(allmp4s):
    name = os.path.basename(mp4)
    os.system(f'ffmpeg -loglevel error -y -i /data3/lipread_test_25/video/{name} {path_wav}/{name[:-4]}.wav')
