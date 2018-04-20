import cv2
import os
import sys
import glob
from classification import identify_faces

def RecogniseFace(input_path,feature,model):
    """
    Function detects and classifies faces in a given image, or in all images in a given folder
    :param input_path: individual jpg file or folder containing jpg files
    :param feature: feature type. Accepted arguments: 'surf', 'hog', 'lbp', 'None'
    :param model: model type. Accepted arguments: 'mlp', 'rf', 'nb', 'svm', 'cnn'
    :return:
    """
    # Check input args ok
    if model=='cnn' and feature !='none':
        print('If using model CNN, feature type must be "None"')
        return

    if model!='cnn' and feature=='none':
        print('Feature type "none" only compatible with model type "cnn"')
        return

    models = ['mlp', 'svm', 'rf', 'nb', 'cnn']
    if model not in models:
        print('Model type "{}" not accepted. Must be one of: {}'.format(model, models))
        return

    feature_types = ['surf', 'lbp', 'hog', 'none']
    if feature not in feature_types:
        print('Feature type "{}" not accepted. Must be one of: {}'.format(feature, feature_types))
        return

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
        feature_type = sys.argv[2].lower()
        classification_method = sys.argv[3].lower()
    else:
        script_dir=''
        input_path=r"C:\Data\computer_vision_coursework\Images\Group11of11\Group11of11\IMG_0613_62.jpg"
        feature_type='surf'
        classification_method='svm'

    if script_dir!='':
        os.chdir(script_dir)

    RecogniseFace(input_path, feature_type, classification_method)
