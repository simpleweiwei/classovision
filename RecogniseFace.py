import cv2
import os
import sys
import glob
from classification import identify_faces

def RecogniseFace(input_path,feature,model):

    results={}
    if not os.path.isdir(input_path):
        # if single file supplied, identify single file and return result
        print("Process file: {}".format(input_path))
        image = cv2.imread(input_path)
        results[input_path] = identify_faces(image, feature, model)
    else:
        # if folder supplied, run for all .jpg and .mov files in folder (much faster since CNN loads only once)
        for file in glob.glob(os.path.join(input_path,'*jpg')):
            print("Process file: {}".format(file))
            image = cv2.imread(file)
            results[file] = identify_faces(image, feature, model)

    print('{ \n    ' + '\n    '.join([k + ':' + str(results[k]) + ',' for k in results])[:-1] + ' \n}')

    return

if __name__ == '__main__':
    #change working directory so that relative paths for saved models work as expected
    # (if any 'file not found' issues, please change to absolute paths in config.py)

    if len(sys.argv) > 1:
        script_dir=os.path.dirname(sys.argv[0])
        input_path = sys.argv[1]
        feature_type = sys.argv[2]
        classification_method = sys.argv[3]
    else:
        script_dir=''
        input_path=r"C:\Users\sarki\Pictures\nicole_kidman.jpg"
        feature_type='None'
        classification_method='cnn'

    if script_dir!='':
        os.chdir(script_dir)

    # input_folder_root=r"U:\Data\computer_vision_coursework\face_images\from_group_photos\val\*"
    # for folder in glob.glob(input_folder_root):
    #     RecogniseFace(folder, feature_type, classification_method)

    RecogniseFace(input_path, feature_type, classification_method)
