import os
import numpy as np
import pickle as pck
import glob
import datetime as dt

from sklearn import svm
from sklearn import metrics,ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from train_classifiers import load_extracted_feature_data

if __name__ == '__main__':

    print('Start testing classifiers')

    feature_data_glob=r'data/extracted_features_augmented_balanced/*npy'
    model_load_path=r'saved_models/augmented_balanced'

    data=load_extracted_feature_data(feature_data_glob)

    for feature_type in data:

        val_features = data[feature_type]['val']['features']
        val_labels = data[feature_type]['val']['labels']

        for model_type in ['nb','rf','svm','mlp']:
            model_file=glob.glob(os.path.join(model_load_path, feature_type + '_' + model_type + '_*pck'))[0]
            print('Loading model {}'.format(model_file))
            mdl2 = pck.load(open(model_file,"rb"))
            mdl2_pred = mdl2.predict(val_features)
            cm = metrics.confusion_matrix(val_labels,mdl2_pred)
            acc = (np.sum(np.diag(cm))/np.sum(cm))
            print('Feature type: {}, Model type: {}, Accuracy: {}'.format(feature_type,model_type,acc))
            #print('{} Confusion matrix for feature type {} using model {}: \n {}'.format(
            #    dt.datetime.strftime(dt.datetime.now(),'%H:%M:%S'), feature_type, model_type, cm)
            #)

    print('Finish testing classifiers')






