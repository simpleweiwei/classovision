import os
import shutil
import glob
import cv2
import numpy as np
import pickle as pck
from keras.models import load_model
import config as cfg

import utils as u
from detection import detect_digits
import feature_extraction as fe

def get_digit_cnn(model_path):

    if not u.is_module_imported('keras'):
        from keras.models import load_model

    return load_model(model_path)

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
        image = cv2.imread(file)
        return identify_digit_from_frame(image, cfg.digit_cnn_path)

def identify_digit_from_video(file):
    cap = cv2.VideoCapture(file, cv2.CAP_FFMPEG)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    success, image = cap.read()

    #use only every 10th frame (peformance reasons)
    nth_frame=cfg.every_n_frames
    frames_to_use=list(range(length))[::nth_frame]

    digit_votes=[]
    for i in frames_to_use:
        cap.set(1, i - 1)
        success, image = cap.read(1)  # image is an array of array of [R,G,B] values
        frameId = cap.get(1)  # The 0th frame is often a throw-away
        digits=identify_digit_from_frame(image,cfg.digit_cnn_path)
        digit_votes.append(digits)

    cap.release()
    #return majority vote of frames
    return u.mode(digit_votes)

def identify_digit_from_frame(image, model_path):
    locs, img, sub_frames = detect_digits(image)

    digit_cnn = load_model(model_path)

    digits = []
    for j, sf in enumerate(sub_frames):
        for k, dg in enumerate(sf[2]):
            digit = classify_individual_digit(digit_cnn, dg)
            digits.append(digit)

    negative_value=get_digit_class_dict()['negatives']

    digit_str = ''.join([str(d) for d in digits if d != negative_value])
    return digit_str

def classify_and_move_digits(model,file):
    img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    pred = classify_individual_digit(model, img)
    okval = 7
    if pred == okval:
        nf = os.path.join(os.path.basename(file),str(okval),os.path.basename(file))
        shutil.move(file,nf)


def get_digit_class_dict():
    class_dict = {}
    class_dict['negatives'] = 10
    for i in range(10):
        class_dict[str(i)] = i
    return class_dict


def classify_face(face_img,method='cnn',feature_type=None, **kwargs):

    model_lookup = {
        'cnn' : {
            None : r'saved_networks/vgg_face61_trained_unaug.h5',
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
        None: do_nothing
    }

    if feature_type == 'surf':
        kwargs={'bow_path':r'data\extracted_features\features_surf_dictsize200_34745_images_BOW_batch_0.npy'}

    if any(np.shape(face_img)) == 0:
        return

    face_img = cv2.resize(face_img, (224, 224))
    model_path = model_lookup[method][feature_type]
    feature_function = feature_lookup[feature_type]

    return model_lookup[method]['function'](face_img, model_path, feature_function, **kwargs)

def do_nothing(face_img):
    return face_img

def classify_face_cnn(face_img, model_path, feature_function, class_labels=[]):
    #class_labels=kwargs['class_labels']
    features = feature_function(face_img)
    features = np.reshape(features, (1,)+np.shape(features))
    model = get_face_cnn(model_path)
    pred = model.predict(features)
    if np.max(pred) < 1:
        print('CNN is less confident about this one, softmax output: \n {}'.format(pred))
        #TODO: remove this comment
    #todo: some nicer logic about returning null if none of the softmaxes are high enough
    pred_ind = classficiation_rule_map(pred)
    return class_labels[pred_ind]


def classify_face_model(face_img, model_path, feature_function, **kwargs):
    features = feature_function(face_img, **kwargs)
    features = np.reshape(features, (1,np.prod(np.shape(features))))
    mdl2 = pck.load(open(model_path, "rb"))
    mdl2_pred = mdl2.predict(features)
    return mdl2_pred

