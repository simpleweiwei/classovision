import os
import numpy as np
import pickle as pck
import glob
import cv2
import datetime as dt
import random
import imutils

from sklearn import svm
from sklearn import metrics,ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from classification import identify_digits_from_file, identify_faces, classify_face
from train_classifiers import load_extracted_feature_data
import utils as u
import config as cfg

def validate_digit_classification(glob_template):
    img_files = glob.glob(glob_template)
    random.shuffle(img_files)
    img_files=img_files[0:1000]

    class_labels=[]
    class_pred=[]
    for i,file in enumerate(img_files):
        if ('mov' not in file.lower()) & ('jpg' not in file.lower()):
            print('File {} of {}: {} not an image or video, skipping'.format(i+1,len(img_files),file))
            continue
        digits = identify_digits_from_file(file)
        class_pred.append(digits)
        corr = u.get_class_label_from_folder(file)
        corr = '11' if 'negatives' in str(corr) else corr
        corr = str(cfg.unknown_face_return_value) if 'unknown' in str(corr) else corr
        corr = corr[1:3] if len(corr)==3 else corr
        class_labels.append(corr)

        print('File {} of {}: {}, Correct label: {}, Prediction: {}'.format(i + 1, len(img_files), file, u.get_class_label_from_folder(file)[1:3], digits))

    class_pred=[x[0] if len(x) > 0 else '' for x in class_pred]
    acc = sum([class_labels[i] in class_pred[i] for i in range(len(class_labels))]) / len(class_labels)
    print('Digit accuracy: {}'.format(acc))
    return acc

def validate_statistical_models(data_dir, model_load_dir):

    feature_data_glob=os.path.join(data_dir,'*npy')

    data=load_extracted_feature_data(feature_data_glob, exclude_strings=['train','cnn'])

    for feature_type in data:

        val_features = data[feature_type]['val']['features']
        val_labels = data[feature_type]['val']['labels']

        for model_type in ['nb','rf','mlp','svm']:
            model_file=glob.glob(os.path.join(model_load_dir, feature_type + '_' + model_type + '_*pck'))[0]
            print('Loading model {}'.format(model_file))
            mdl2 = pck.load(open(model_file,"rb"))
            mdl2_pred = mdl2.predict(val_features)
            #cm = metrics.confusion_matrix(val_labels,mdl2_pred)
            acc = metrics.accuracy_score(val_labels,mdl2_pred)
            #per_class_precision=[round(x,2) for x in metrics.precision_score(val_labels,mdl2_pred, average=None)]
            print('Feature type: {}, Model type: {}, Accuracy: {}'.format(feature_type,model_type,acc))

    return

def validate_face_recognition(val_data_dir,feat_type='None',model='cnn'):

    test_files = glob.glob(os.path.join(val_data_dir,'*\*jpg'))
    random.shuffle(test_files)
    test_files = test_files[0:1000]

    prediction=[]
    correct_label=[]
    for i,file in enumerate(test_files):
        image = cv2.imread(file)
        #check if image has been rotated and rotate back
        h,w,c=np.shape(image)
        if (h < 2000) & (h < w):
            image = cv2.transpose(image)
            image = cv2.flip(image, 1)

        pred = identify_faces(image, feat_type, model)
        try:
            prediction.append(list(pred[:,0]))
        except IndexError:
            prediction.append([])
        corr = u.get_class_label_from_folder(file)
        corr = cfg.unknown_face_return_value if 'unknown' in corr else int(corr[1:3])
        correct_label.append(corr)
        print('File: {}, Size: {}, Correct: {}, Prediction: {}'.format(file, np.shape(image), corr, pred))

    acc = sum([correct_label[i] in prediction[i] for i,v in enumerate(correct_label)])/len(correct_label)
    print('CNN accuracy: {}'.format(acc))
    return

def validate_face_cnn_on_face_images(face_image_dir):

    validation_images = glob.glob(os.path.join(face_image_dir,'045\*.jpg'))
    random.shuffle(validation_images)
    validation_images=validation_images

    prediction=[]
    correct_label=[]
    for val_img in validation_images:
        image = cv2.imread(val_img)
        if (image is None) or (any([x==0 for x in np.shape(image)])):
            continue

        corr = u.get_class_label_from_folder(val_img)
        corr = cfg.unknown_face_return_value if 'unknown' in corr else corr
        correct_label.append(corr)
        pred = classify_face(image, method='cnn', feature_type='None')
        prediction.append(pred)

        print('File: {}, Correct: {}, Prediction: {}'.format(val_img, corr, pred))

    acc = sum([correct_label[i] == prediction[i] for i, v in enumerate(correct_label)]) / len(correct_label)
    cm = metrics.confusion_matrix(correct_label,prediction, labels=cfg.class_labels)
    print('CNN Accuracy: {} \n CNN Confusion matrix: \n {}'.format(acc,cm))

    return

def validate_group_photo_performance(group_photo_dir,feat_type='None',model='cnn'):

    file_glob=os.path.join(group_photo_dir,'*.jpg')
    file_list=glob.glob(file_glob)

    for file in file_list:
        image = cv2.imread(file)
        print('File {} has size: {}'.format(file,np.shape(image)))
        faces = identify_faces(image, feat_type, model)
        for i in range(np.shape(faces)[0]):
            cv2.circle(image, (faces[i,1], faces[i,2]), 10, (255, 255, 0), -11)
            cv2.putText(image, str(faces[i,0]), (faces[i, 1] + 10, faces[i, 2] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, round(np.shape(image)[1]/2000),
                        (255, 255, 255), 2)

        image = imutils.resize(image,width=1500)
        u.imshow(image)

    return

if __name__ == '__main__':

    print('Start testing classifiers')

    #validate_statistical_models(r'data/extracted_features', 'saved_models')
    #validate_face_recognition(r'C:\Data\computer_vision_coursework\Images\individual_people\val')
    #validate_group_photo_performance(r'C:\Data\computer_vision_coursework\Images\original_images\11of11\Group11of11')
    #validate_digit_classification(r'C:\Data\computer_vision_coursework\Images\original_images\*\*\*\*')
    #validate_digit_classification(r'C:\Data\computer_vision_coursework\Images\original_images\8of11\8of11\059\IMG_0307.JPG')

    validate_face_cnn_on_face_images(r'C:\Data\computer_vision_coursework\Images\face_images\from_individual_photos\val')

    #digit_train=r'C:\Data\computer_vision_coursework\Images\digit_images\ind_digits\train'
    #digit_val=r'C:\Data\computer_vision_coursework\Images\digit_images\ind_digits\val'
    #train_frac=0.1
    #u.move_from_train_to_val(digit_train, digit_val, train_frac)

    #validate_digit_classification(r'C:\Data\computer_vision_coursework\Images\digit_images\ind_digits\val\*\*jpg')
    #validate_digit_classification(r'C:\Data\computer_vision_coursework\Images\individual_people\val\*\*jpg')

    print('Finish testing classifiers')






