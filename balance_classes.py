import imgaug
import os
import glob
import cv2
import numpy as np
import random
import shutil

from_folder=r"U:\Data\computer_vision_coursework\face_images\from_both\augmented"
to_folder=r"U:\Data\computer_vision_coursework\face_images\from_both\augmented_balanced"
from_glob=os.path.join(from_folder,'*')
folders_to_check=glob.glob(from_glob)

target_obs=800

print_no=0
for folder in folders_to_check:
    print('Starting folder: {}'.format(os.path.basename(folder)))

    file_glob=os.path.join(folder,'*jpg')
    file_list=glob.glob(file_glob)
    random.shuffle(file_list)
    file_list=file_list[0:target_obs]

    target_folder = os.path.join(to_folder, os.path.basename(folder))
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

    for file in file_list:
        target_file=os.path.join(target_folder,os.path.basename(file))
        #shutil.copy(file,target_file)
        print('Copied file from {} to {}'.format(file,target_file))


print('done!')