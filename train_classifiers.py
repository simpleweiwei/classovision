import os
import numpy as np
import pickle as pck
import glob
import datetime as dt

from sklearn import svm
from sklearn import metrics,ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

np.set_printoptions(edgeitems=20,linewidth=200)

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

def load_extracted_feature_data(glob_path,exclude_strings=[]):

    feature_files = glob.glob(glob_path)
    feature_files = [x for x in feature_files if 'BOW' not in x.upper()]
    feature_files = [x for x in feature_files if all([x.upper().find(s.upper())<0 for s in exclude_strings]) ]

    agg_data = {}

    for feat_file in feature_files:
        print('Loading input file {}...'.format(feat_file))
        batch_result = np.load(feat_file)[()]
        ft = batch_result['feature_type']
        set = batch_result['set']
        feature_label = [x for x in batch_result.keys() if 'features' in x][0]
        label_label = [x for x in batch_result.keys() if 'label' in x][0]
        if ft not in list(agg_data.keys()):
            agg_data[ft] = {}
        if set not in list(agg_data[ft].keys()):
            agg_data[ft][set] = {}

        cat_dict = {'features':feature_label,'labels':label_label}
        for k in cat_dict:
            if k not in agg_data[ft][set].keys():
                #if this is the first data batch of these filter keys to be added, just add it
                agg_data[ft][set][k] = batch_result[cat_dict[k]]
            else:
                #otherwise append to existing
                agg_data[ft][set][k] = np.concatenate([agg_data[ft][set][k],batch_result[cat_dict[k]]],axis=0)

    return agg_data

if __name__ == '__main__':

    print('Process start: {}'.format(dt.datetime.strftime(dt.datetime.now(),'%H:%M:%S')))
    # new load method adapted to accept batches of data
    saved_feature_path=r'data/extracted_features_augmented_balanced/new_surf_dict'
    file_pattern = 'features_*_*_images_*_batch_*.npy'
    glob_path = os.path.join(saved_feature_path, file_pattern)

    model_save_location = r'./saved_models/augmented_balanced'
    #model_save_location=r'D:\augmented_balanced'

    results=load_extracted_feature_data(glob_path, exclude_strings=['train'])

    for feature_type in results:

        train_features = results[feature_type]['train']['features']
        train_labels = results[feature_type]['train']['labels']

        val_features = results[feature_type]['val']['features']
        val_labels = results[feature_type]['val']['labels']

        function_dict = {
            'rf':train_rf,
            'mlp':train_mlp,
            'nb': train_nb,
            'svm': train_svm
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

    """

    #get unique from files for save
    import pandas as pd
    df = pd.DataFrame(results['surf']['train']['features'])
    df['label'] = results['surf']['train']['labels']
    counts=df.drop_duplicates().groupby('label')[[0]].count()

    train_df = df.drop_duplicates()
    train_labels = train_df['label'].as_matrix()
    feature_cols=[x for x in train_df.columns if type(x) != str]
    train_features = train_df[feature_cols].as_matrix()

    temp_results=r'./data/temp_extracted_features_augmented_balanced'
    if not os.path.isdir(temp_results):
        os.makedirs(temp_results)

    new_results = {}
    new_results['surf'] = {}
    new_results['surf']['train'] = {}
    new_results['surf']['train']['features'] = train_features
    new_results['surf']['train']['labels'] = train_labels

    set='train'
    save_batch_size=1000
    rows, cols = np.shape(new_results['surf'][set]['features'])

    batches = ceil(rows / save_batch_size)
    for b in range(batches):
        first = b * save_batch_size
        last = min((1 + b) * save_batch_size, rows)
        feats = new_results[ft][set + '_features'][first:last]
        new_results[ft][set + '_features_' + str(b)] = feats
        labs = new_results[ft][set + '_labels'][first:last]
        new_results[ft][set + '_labels_' + str(b)] = labs
        new_results[ft]['set'] = set
        save_nam_batch = 'features_surf_dictsize200_{}_images_train_filebatch_{}_batch_{}.npy'.format(rows,0,b)
        np.save(os.path.join(temp_results, save_nam_batch), new_results)

    val_folder=r"U:\Data\computer_vision_coursework\face_images\from_both\val\*"
    val_folders=glob.glob(val_folder)
    for valf in val_folders:
        label=os.path.basename(valf)
        glob_path=os.path.join(valf,'*.jpg')
        val_files_expected=glob.glob(glob_path)
        counts.ix[label,'expected'] = len(val_files_expected)
    """
    print('Done!')