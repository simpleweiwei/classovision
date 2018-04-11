import os
import numpy as np
import pickle as pck
import glob
import datetime as dt

from sklearn import svm
from sklearn import metrics,ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

def train_nb(train_features,train_labels):
    gnb = GaussianNB()
    gnb.fit(train_features, train_labels)
    return gnb

def train_mlp(train_features,train_labels):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (30,30,30), random_state = 1)
    clf.fit(train_features, train_labels)
    return clf

def train_svm(train_features,train_labels):
    clf = svm.SVC()
    clf.fit(train_features,train_labels)
    return clf

def train_rf(train_features,train_labels):
    clf=ensemble.RandomForestClassifier()
    clf.fit(train_features,train_labels)
    return clf



if __name__ == '__main__':

    print('Process start: {}'.format(dt.datetime.strftime(dt.datetime.now(),'%H:%M:%S')))
    # new load method adapted to accept batches of data
    saved_feature_path=r'data/extracted_features'
    file_pattern = 'features_*_*_images_*_batch_*.npy'
    feature_files = glob.glob(os.path.join(saved_feature_path,file_pattern))
    feature_files = [x for x in feature_files if 'BOW' not in x.upper()]
    feature_files = [x for x in feature_files if 'CNN' not in x.upper()]
    feature_files = [x for x in feature_files if 'HOG' in x.upper()]
    model_save_location=r'./saved_models'

    results = {}

    for feat_file in feature_files:
        print('Loading input file {}...'.format(feat_file))
        batch_result = np.load(feat_file)[()]
        ft = batch_result['feature_type']
        set = batch_result['set']
        feature_label = [x for x in batch_result.keys() if 'features' in x][0]
        label_label = [x for x in batch_result.keys() if 'label' in x][0]
        if ft not in list(results.keys()):
            results[ft] = {}
        if set not in list(results[ft].keys()):
            results[ft][set] = {}

        cat_dict = {'features':feature_label,'labels':label_label}
        for k in cat_dict:
            if k not in results[ft][set].keys():
                #if this is the first data batch of these filter keys to be added, just add it
                results[ft][set][k] = batch_result[cat_dict[k]]
            else:
                #otherwise append to existing
                results[ft][set][k] = np.concatenate([results[ft][set][k],batch_result[cat_dict[k]]],axis=0)

    for feature_type in results:

        train_features = results[feature_type]['train']['features']
        train_labels = results[feature_type]['train']['labels']

        val_features = results[feature_type]['val']['features']
        val_labels = results[feature_type]['val']['labels']

        function_dict = {
            #'rf':train_rf,
            #'mlp':train_mlp
            'svm': train_svm
            #'nb': train_nb
        }

        for model in function_dict:
            print('{} - Training model {} on training features of dimensions {}...'.format(dt.datetime.strftime(dt.datetime.now(),'%H:%M:%S'),model,np.shape(train_features)))
            mdl = function_dict[model](train_features,train_labels)
            model_save_name = '{}_{}_trained_{}obs.pck'.format(feature_type,model,len(train_labels))
            model_save_path = os.path.join(model_save_location,model_save_name)
            pck.dump(mdl, open(model_save_path, "wb"))
            print('{} Saved classifier: {}'.format(dt.datetime.strftime(dt.datetime.now(),'%H:%M:%S'), model_save_path))
            mdl2 = pck.load(open(model_save_path,"rb"))
            mdl2_pred = mdl2.predict(val_features)

            cm = metrics.confusion_matrix(val_labels,mdl2_pred)
            print('{} Confusion matrix for feature type {} using model {}: \n {}'.format(dt.datetime.strftime(dt.datetime.now(),'%H:%M:%S'), feature_type,model,cm))

    print('Done!')