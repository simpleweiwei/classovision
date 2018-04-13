import os
import numpy as np
import pickle as pck
import glob
import datetime as dt

from sklearn import svm
from sklearn import metrics,ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from classification import identify_digits_from_file
from train_classifiers import load_extracted_feature_data
import utils as u

def test_digit_classification(glob_template):
    img_files = glob.glob(glob_template)

    model_path=r'saved_networks\cnn_alldignan.h5'
    #digit_cnn = load_model(r'saved_networks\cnn_alldignan.h5')

    class_labels=[]
    class_pred=[]
    for i,file in enumerate(img_files):
        digits = identify_digits_from_file(file)
        class_labels.append(u.get_class_label_from_folder(file)[1:3])
        class_pred.append(digits)
        print('File {} of {}: {}, Correct label: {}, Prediction: {}'.format(i + 1, len(img_files), file, u.get_class_label_from_folder(file)[1:3], digits))

    class_pred=[x[0] if len(x) > 0 else '' for x in class_pred]
    acc = sum([class_labels[i] == class_pred[i] for i in range(len(class_labels))]) / len(class_labels)
    print('Digit accuracy: {}'.format(acc))
    return acc

if __name__ == '__main__':

    print('Start testing classifiers')

    feature_data_glob=r'data/extracted_features_augmented_balanced/*npy'
    model_load_path=r'saved_models/augmented_balanced'

    data=load_extracted_feature_data(feature_data_glob)

    for feature_type in data:

        val_features = data[feature_type]['val']['features']
        val_labels = data[feature_type]['val']['labels']

        for model_type in ['nb','rf','mlp']:
            model_file=glob.glob(os.path.join(model_load_path, feature_type + '_' + model_type + '_*pck'))[0]
            print('Loading model {}'.format(model_file))
            mdl2 = pck.load(open(model_file,"rb"))
            mdl2_pred = mdl2.predict(val_features)
            #cm = metrics.confusion_matrix(val_labels,mdl2_pred)
            acc = metrics.accuracy_score(val_labels,mdl2_pred)
            per_class_precision=metrics.precision_score(val_labels,mdl2_pred, average=None)
            print('Feature type: {}, Model type: {}, Accuracy: {}'.format(feature_type,model_type,acc))

    print('Finish testing classifiers')






