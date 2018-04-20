"""
Script uses clustering to assist with face dataset generation: used to help group individuals' face images
"""

import numpy as np
import sys
import os
import datetime as dt
import glob
import shutil
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

if __name__ == '__main__':
    labels_path=r"C:\Data\computer_vision_coursework\Images\Group11of11\Group11of11\extracted_faces\*jpg"
    glob_path=r"C:\Data\computer_vision_coursework\Images\Group11of11\Group11of11\extracted_faces\feature_data\features_cnn_10178_images_train_batch_*.npy"
    feature_files = glob.glob(glob_path)
    for file in feature_files:
        batch_result = np.load(file)[()]
        train_key = [x for x in batch_result.keys() if 'features' in x][0]

        if 'features' not in locals():
            features = batch_result[train_key]
        else:
            features = np.concatenate([features, batch_result[train_key]], axis=0)

    n_samples, n_dims = np.shape(features)
    n_dims_new=500
    pca = PCA(n_components=n_dims_new)
    print('Running PCA to reduce from {} to {} dimensions for {} samples... {}'.format(n_dims, n_dims_new, n_samples, dt.datetime.strftime(dt.datetime.now(),'%H:%M:%S')))
    X = pca.fit_transform(features)
    print('PCA done, {}'.format(dt.datetime.strftime(dt.datetime.now(),'%H:%M:%S')))

    # now do kmeans
    #print('Starting Kmeans with {} samples on {} dimensions... {}'.format(n_samples, n_dims_new, dt.datetime.strftime(dt.datetime.now(),'%H:%M:%S')))
    #kmeans = KMeans(n_clusters=200, random_state=0).fit(X)
    #print('KMeans done, {}'.format(dt.datetime.strftime(dt.datetime.now(), '%H:%M:%S')))
    #print(kmeans.labels_)
    #cluster_labels = kmeans.labels_

    from sklearn.neighbors import kneighbors_graph

    print('Starting k nearest neighbours graph... {}'.format(dt.datetime.strftime(dt.datetime.now(),
                                                                                               '%H:%M:%S')))
    X = features
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    print('Starting agglomerative clustering... {}'.format(dt.datetime.strftime(dt.datetime.now(),
                                                                                               '%H:%M:%S')))
    ward = AgglomerativeClustering(n_clusters=150, connectivity=connectivity,
                                   linkage='ward').fit(X)
    cluster_labels = ward.labels_

    orig_files = glob.glob(labels_path)
    file_labels = [os.path.basename(f) for f in orig_files]
    cluster_dict = dict(zip(orig_files,cluster_labels))

    print('Begin file copy assignment... {}'.format(dt.datetime.strftime(dt.datetime.now(), '%H:%M:%S')))
    new_file_dir = r'C:\Data\computer_vision_coursework\Images\Group11of11\Group11of11\extracted_faces\clustering_4_aggl_alldim'
    if not os.path.isdir(new_file_dir):
        os.mkdir(new_file_dir)

    for file in cluster_dict:
        new_file_name = os.path.basename(file).replace('group_face','face_cluster{}'.format(cluster_dict[file]))
        new_path = os.path.join(new_file_dir,new_file_name)
        shutil.copy(file,new_path)
    print('End file copy assignment... {}'.format(dt.datetime.strftime(dt.datetime.now(), '%H:%M:%S')))



    print('done!')