import os
import pickle as pck
import shutil

import cv2
import numpy as np

import config as cfg
import feature_extraction as fe
import utils as u
from detection import detect_digits, detect_faces
from utils import get_digit_class_dict


def get_digit_cnn(model_path):

    if 'digit_cnn_model' in globals():
        return globals()['digit_cnn_model']
    else:
        global digit_cnn_model
        if not u.is_module_imported('keras'):
            from keras.models import load_model
        digit_cnn_model = load_model(model_path)
        return digit_cnn_model

def get_face_cnn(model_path):

    if 'face_cnn_model' in globals():
        return globals()['face_cnn_model']
    else:
        global face_cnn_model
        from keras.models import load_model
        face_cnn_model = load_model(model_path)
        return face_cnn_model

def classficiation_rule_map(prediction_vector):
    return np.argmax(prediction_vector)

def classify_individual_digit(model, digit_img):
    digit_reshaped = digit_img.reshape(1, 28, 28, 1)
    digit_predictions = model.predict(digit_reshaped)
    result = classficiation_rule_map(digit_predictions)
    return result

def identify_digits_from_file(file):
    if u.file_is_video(file):
        return identify_digit_from_video(file)
    else:
        if u.is_error_image(file):
            return "Error loading image"

        image = cv2.imread(file)
        return identify_digit_from_frame(image, cfg.digit_cnn_path)

def identify_digit_from_video(file):
    cap = cv2.VideoCapture(file, cv2.CAP_FFMPEG)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    success, image = cap.read()

    #use only every nth frame (peformance reasons)
    frame_frac=cfg.frame_frac
    nth_frame=int(length*frame_frac)
    frames_to_use=list(range(length))[::nth_frame]

    digit_votes=[]
    for i in frames_to_use:
        cap.set(1, i - 1)
        success, image = cap.read(1)  # image is an array of array of [R,G,B] values
        frameId = cap.get(1)  # The 0th frame is often a throw-away

        if len(np.shape(image)) != 3:
            continue

        #rotate image if landscape
        if np.shape(image)[0] < np.shape(image)[1]:
            image = cv2.transpose(image)
            image = cv2.flip(image, 1)

        digits=identify_digit_from_frame(image,cfg.digit_cnn_path)
        digit_votes.append(digits)

    cap.release()
    #return majority vote of frames
    majority_vote = u.mode_of_2digit_strings(digit_votes)

    return majority_vote

def identify_digit_from_frame(image, model_path):
    """
    Function detects and classifies numbers from a single image frame. Where multiple numbers are detected, return all in a list
    :param image: image numpy array
    :param model_path: path to cnn model used to classify
    :return: list of detected numbers
    """
    from keras.models import load_model
    #get sub-frames of number ROIs
    sub_frames = detect_digits(image,sharpen=True, debug=False)
    digit_cnn = load_model(model_path)
    negative_value = get_digit_class_dict()['negatives']

    all_digit_strings_found=[]
    #each sub_frame is a list of digit images
    for j, sf in enumerate(sub_frames):
        digits=[]
        for k, dg in enumerate(sf):
            digit = classify_individual_digit(digit_cnn, dg)
            if digit != negative_value:
                digits.append(str(digit))

        #concatenate all digits within each location
        if len(digits)==2:
            digit_str = ''.join([str(d) for d in digits])
            all_digit_strings_found.append(digit_str)

    return all_digit_strings_found

def classify_face(face_img,method='cnn',feature_type='None', **kwargs):

    model_lookup = {
        'cnn' : {
            'none' : cfg.face_cnn,
            'function' : classify_face_cnn
        },
        'svm' : {
            'hog' : cfg.svm_hog,
            'surf' : cfg.svm_surf,
            'lbp' : cfg.svm_lbp,
            'function' : classify_face_model
        },
        'mlp' : {
            'hog' : cfg.mlp_hog,
            'surf' : cfg.mlp_surf,
            'lbp': cfg.mlp_lbp,
            'function' : classify_face_model
        },
        'rf' : {
            'hog' : cfg.rf_hog,
            'surf' : cfg.rf_surf,
            'lbp': cfg.rf_lbp,
            'function' : classify_face_model
        },
        'nb' : {
            'hog' : cfg.nb_hog,
            'surf' : cfg.nb_surf,
            'lbp': cfg.nb_lbp,
            'function' : classify_face_model
        }
    }

    feature_lookup = {
        'hog' : fe.get_hog_features,
        'surf' : fe.get_surfbow_features,
        'lbp' : fe.get_lbp_features,
        'cnn' : fe.get_cnn_features,
        'none': do_nothing
    }

    if feature_type == 'surf':
        kwargs['bow_path'] = cfg.bow_file

    if any(np.shape(face_img)) == 0:
        return

    face_img = cv2.resize(face_img, (224, 224))
    model_path = model_lookup[method][feature_type]
    feature_function = feature_lookup[feature_type]

    return model_lookup[method]['function'](face_img, model_path, feature_function, **kwargs)

def do_nothing(face_img):
    return face_img

def classify_face_cnn(face_img, model_path, feature_function):
    """ Classify face using CNN """
    class_labels = cfg.class_labels
    features = feature_function(face_img)
    features = np.reshape(features, (1,)+np.shape(features))
    model = get_face_cnn(model_path)
    pred = model.predict(features)
    pred_ind = classficiation_rule_map(pred)
    return class_labels[pred_ind]

def classify_face_model(face_img, model_path, feature_function, **kwargs):
    """ Classify face using ML model with separate feature extraction"""
    features = feature_function(face_img, **kwargs)
    features = np.reshape(features, (1,np.prod(np.shape(features))))
    mdl2 = pck.load(open(model_path, "rb"))
    mdl2_pred = mdl2.predict(features)
    return mdl2_pred[0]

def identify_faces(image, feature_type, classifier_name):
    """ Function detects and classifies faces in an image """

    model_file = cfg.ssd_model
    prototxt_file = cfg.prototxt_file

    if np.shape(image)[0] > np.shape(image)[1]:
        #individual photo config
        merge_overlap = 0.1
        aspect_ratio_bounds = (0.4, 2)
        min_confidence = 0.6
        step_size = 1000
        window_size = (2000, 2000)
    else:
        # group photo config
        merge_overlap = 0.2
        aspect_ratio_bounds = (0.5, 1.7)
        min_confidence = 0.5
        ws = int(round(np.shape(image)[0]/6))
        step_size = int(round(ws/2))
        window_size = (ws, ws)

    face_bboxes = detect_faces(image, model_file, prototxt_file, min_confidence, aspect_ratio_bounds, merge_overlap,
                               step_size, window_size)

    for face_bbox in face_bboxes:
        x1, y1, x2, y2 = face_bbox
        face_img = image[y1:y2, x1:x2]

        #skip empty bboxes
        if any([x==0 for x in np.shape(face_img)]):
            continue
        x = int(round(np.mean([x1,x2])))
        y = int(round(np.mean([y1, y2])))
        face_id = classify_face(face_img, method=classifier_name, feature_type=feature_type)
        #u.imshow(face_img)
        #remove unknowns and convert to 2-character ID
        face_id = int(face_id) if 'unknown' not in face_id else cfg.unknown_face_return_value
        this_result=np.array([[face_id,x,y]],dtype=np.int64)

        #concat to results matrix if it exists, otherwise create it
        if 'result_mat' not in locals():
            result_mat = this_result
        else:
            result_mat = np.concatenate([result_mat,this_result],axis=0)

    #if no faces found, return empty array
    if len(face_bboxes)==0 or 'result_mat' not in locals():
        #if no faces detected or all face bboxes detected invalid (1 dim ==0)
        result_mat=np.array([[]])

    return result_mat
