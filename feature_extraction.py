import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import spatial
import glob
import os

import pickle as pck
from math import ceil,floor
import random
from skimage import feature

#import utils as u

def is_error_image(image_file):
    try:
        image = cv2.imread(image_file)
    except:
        return True

    if image is None:
        return True
    else:
        return False

def get_class_label_from_folder(image_file):
    return os.path.basename(os.path.dirname(image_file))

def get_class_labels_from_folders(image_files):
    class_labels=[]
    for img_file in image_files:
        class_label = get_class_label_from_folder(img_file)
        class_labels.append(class_label)
    return class_labels


def get_features_for_image_list(image_list,feature_type='hog',surf_book_of_words=None):

    func_lookup = {
        'hog':get_features_for_image_list_hog,
        'cnn':get_features_for_image_list_cnn,
        'surf':get_features_for_image_list_surf,
        'lbp':get_features_for_image_list_lbp
    }

    if feature_type=='surf':
        add_kwargs={'bow_codebook':surf_book_of_words}
    else:
        add_kwargs={}

    return func_lookup[feature_type](image_list,**add_kwargs)

def get_features_for_image_list_surf(image_list,bow_codebook):
    image_desc,image_labels = get_image_descriptions_surf(image_list)
    if bow_codebook is None:
        print('No Existing Book of Words supplied to generate SURF features, generating BOW from provided image list')
        bow_codebook = get_bow(image_desc)
    else:
        print('Generating Vocab Histogram using Book of Words provided')
    vocab_hist = get_vocab_hist(image_desc, bow_codebook)
    return (vocab_hist,image_labels)

def load_bag_of_words(glob_path):

    bow_files = glob.glob(glob_path)
    for bow_file in bow_files:
        batch_result = np.load(bow_file)[()]

        if 'bag_of_words' not in locals():
            bag_of_words = batch_result
        else:
            bag_of_words = np.concatenate([bag_of_words,batch_result], axis=0)

    return bag_of_words

def get_surfbow_features(image, bow_path, hessian_threshold=100, use_usurf=False, use_extended=True):
    #get image descriptions
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    kp, dsc = get_surf_features(gray, hessian_threshold=hessian_threshold, use_usurf=use_usurf,
                                use_extended=use_extended)

    bag_of_words = load_bag_of_words(bow_path)

    vocab_hist = get_vocab_hist([dsc],bag_of_words)

    return vocab_hist

def get_features_for_image_list_lbp(image_list):
    image_labels=[]
    for i,img_file in enumerate(image_list):
        try:
            image = cv2.imread(img_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            features = get_lbp_features(image)
        except Exception as ex:
            print('Error generating LBP features for file {}, bypassing file'.format(img_file))
            continue

        image_labels.append(get_class_label_from_folder(img_file))

        if not 'feature_mat' in locals():
            feature_mat = np.zeros((len(image_list), np.prod(np.shape(features))))

        feature_mat[i] = np.reshape(features,(1,np.prod(np.shape(features))))

    return (feature_mat,image_labels)

def get_features_for_image_list_hog(image_list,cell_size=(8,8),block_size=(2,2),nbins=9):
    image_labels=[]
    for i,img_file in enumerate(image_list):
        try:
            image = cv2.imread(img_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            features = get_hog_features(image,cell_size,block_size,nbins)
        except Exception as ex:
            print('Error generating HOG features for file {}, bypassing file'.format(img_file))
            continue

        image_labels.append(get_class_label_from_folder(img_file))

        if not 'feature_mat' in locals():
            feature_mat = np.zeros((len(image_list), np.prod(np.shape(features))))

        feature_mat[i] = np.reshape(features,(1,np.prod(np.shape(features))))

    return (feature_mat,image_labels)

def get_features_for_image_list_cnn(image_list):
    image_list = [x for x in image_list if not is_error_image(x)]
    image_labels = get_class_labels_from_folders(image_list)
    for i,img_file in enumerate(image_list):
        image = cv2.imread(img_file)

        if not 'img_mat' in locals():
            img_mat = np.zeros((len(image_list),) + np.shape(image))

        img_mat[i] = image

    feature_mat = get_cnn_features(img_mat)

    return (feature_mat,image_labels)


def get_lbp_features(image, eps=1e-7, num_points=24, radius=8):
    #https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns

    if len(np.shape(image)) > 2:
        if np.shape(image)[2] > 1:
            image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(image, num_points,
                                       radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, num_points + 3),
                             range=(0, num_points + 2))

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    # return the histogram of Local Binary Patterns
    return hist


def get_hog_features(image,cell_size=(8,8),block_size=(2,2),nbins=9):
    #https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python
    #cell_size = (8, 8)  # h x w in pixels
    #block_size = (2, 2)  # h x w in cells
    #nbins = 9  # number of orientation bins
    image = image.copy()
    image = cv2.resize(image,(80,80))
    # winSize is the size of the image cropped to an multiple of the cell size
    hog = cv2.HOGDescriptor(_winSize=(image.shape[1] // cell_size[1] * cell_size[1],
                                      image.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    n_cells = (image.shape[0] // cell_size[0], image.shape[1] // cell_size[1])
    hog_feats = hog.compute(image)\
                   .reshape(n_cells[1] - block_size[1] + 1,
                            n_cells[0] - block_size[0] + 1,
                            block_size[0], block_size[1], nbins) \
                   .transpose((1, 0, 2, 3, 4))  # index blocks by rows first
    # hog_feats now contains the gradient amplitudes for each direction,
    # for each cell of its group for each group. Indexing is by rows then columns.

    gradients = np.zeros((n_cells[0], n_cells[1], nbins))

    # count cells (border cells appear less often across overlapping groups)
    cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

    for off_y in range(block_size[0]):
        for off_x in range(block_size[1]):
            gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                      off_x:n_cells[1] - block_size[1] + off_x + 1] += \
                hog_feats[:, :, off_y, off_x, :]
            cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                       off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

    # Average gradients
    gradients /= cell_count
    return gradients

def get_cnn_features(image_data,batch_size=50):
    #note: make sure you pass in an RGB image
    #TODO: double check if this network wants RGB or BGR
    #reshape to dimensions to pass into CNN
    shp = np.shape(image_data)
    if shp[0:2] == (224, 224):
        # if image_data refers to a single image, add a first dimension to set n=1
        image_data = np.reshape(image_data,(1,)+ shp)

    #load CNN without top layers to get only feature extraction layers (512 filters)
    from keras.applications import VGG16
    weights=r'saved_networks/vggface_headless_weights.h5'
    vgg_conv = VGG16(weights=weights, include_top=False, input_shape=(224, 224, 3))

    #batch_size = 20
    n_samples = np.shape(image_data)[0]

    #gen output array size for extracted features
    features = np.zeros(shape=(n_samples, 7, 7, 512))
    for i in range(ceil(n_samples/batch_size)):
        batch_start=i * batch_size
        batch_end=min((i + 1) * batch_size,n_samples)
        print('Generating CNN features for batch: [{}:{}]'.format(batch_start,batch_end))
        image_batch = image_data[batch_start:batch_end]
        features_batch = vgg_conv.predict(image_batch)
        features[batch_start:batch_end] = features_batch

    features = np.reshape(features,(n_samples,)+(np.prod(np.shape(features)[1:]),))

    return features


def get_cnn_features_imagenet(image_data,batch_size=10):
    #note: make sure you pass in an RGB image
    #TODO: double check if this network wants RGB or BGR
    #reshape to dimensions to pass into CNN
    shp = np.shape(image_data)
    if shp[0:2] == (224, 224):
        # if image_data refers to a single image, add a first dimension to set n=1
        image_data = np.reshape(image_data,(1,)+ shp)

    #load CNN without top layers to get only feature extraction layers (512 filters)
    from keras.applications import VGG16
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    #batch_size = 20
    n_samples = np.shape(image_data)[0]

    #gen output array size for extracted features
    features = np.zeros(shape=(n_samples, 7, 7, 512))
    for i in range(ceil(n_samples/batch_size)):
        batch_start=i * batch_size
        batch_end=min((i + 1) * batch_size,n_samples)
        print('Generating CNN features for batch: [{}:{}]'.format(batch_start,batch_end))
        image_batch = image_data[batch_start:batch_end]
        features_batch = vgg_conv.predict(image_batch)
        features[batch_start:batch_end] = features_batch

    features = np.reshape(features,(n_samples,)+(np.prod(np.shape(features)[1:]),))

    return features

def get_surf_features(image, hessian_threshold=400, use_usurf=False, use_extended=True):
    #https://docs.opencv.org/3.4.0/df/dd2/tutorial_py_surf_intro.html
    # Create SURF object. You can specify params here or later.
    # Here I set Hessian Threshold to 400
    try:
        import cv2
        surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    except Exception as ex:
        print(r"ERROR: {} \n SIFT and SURF are not included in OpenCV 3.4 by default, you can try: 'pip install opencv-contrib-python'".format(ex))

    #find U-SURF to ignore orientation
    # Check upright flag, if it False, set it to True
    surf.setUpright(use_usurf)

    surf.setExtended(use_extended)
    kp, des = surf.detectAndCompute(image, None)
    #print('SURF found {} keypoints using hessian threshold {} '.format(len(kp), surf.getHessianThreshold()))
    return kp,des

def get_image_descriptions_surf(image_files,hessian_threshold=100, use_usurf=False, use_extended=True):
    print('Calculating SURF image descriptions for {} images'.format(len(image_files)))
    image_descriptions=[]
    image_labels=[]
    for p in image_files:
        try:
            image = cv2.imread(p)
            gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        except Exception as ex:
            print('Unable to read or grayscale file {} \n Error: {}'.format(p,ex))
            continue
        kp, dsc= get_surf_features(gray,hessian_threshold=hessian_threshold,use_usurf=use_usurf,use_extended=use_extended)
        image_descriptions.append(dsc)
        image_labels.append(get_class_label_from_folder(p))

    return (image_descriptions,image_labels)

def get_bow_from_image_list(image_list,dict_size):
    image_desc = get_image_descriptions_surf(image_list)[0]
    return get_bow(image_desc, dict_size=dict_size)

def get_bow(image_descriptions,dict_size=1000):
    #https://stackoverflow.com/questions/33713939/python-bag-of-words-clustering

    BOW = cv2.BOWKMeansTrainer(dict_size)

    for dsc in image_descriptions:
        BOW.add(dsc)

    #dictionary created
    print('Bag of Words gen: clustering with k={}'.format(dict_size))
    dictionary = BOW.cluster()
    print('Bag of Words clustering complete')
    return dictionary

def get_vocab_hist(image_descriptions,bag_of_words):
    #https://kushalvyas.github.io/BOV.html
    print('Calculating vocabulary histogram for {} image descriptions'.format(len(image_descriptions)))
    n_clusters=np.shape(bag_of_words)[0]
    n_images=len(image_descriptions)
    vocab_hist=np.array([np.zeros(n_clusters) for i in range(n_images)])

    #for i in range(n_images):
    for i,im_desc in enumerate(image_descriptions):
        print('Generating histogram for image {} of {}'.format(i+1,n_images))
        if type(im_desc) != np.ndarray:
            continue

        l = len(im_desc)
        for j in range(l):
            # for each description, find the BOW centroid it is closest to
            dsc = im_desc[j]
            clust_idx = spatial.KDTree(bag_of_words).query(dsc)[1]
            vocab_hist[i][clust_idx] += 1

    return vocab_hist

if __name__ == '__main__':
    print('Feature extraction process start')

    train_template = r'U:\Data\computer_vision_coursework\face_images\from_both\augmented_balanced\*\*.jpg'
    training_images = glob.glob(train_template)
    #random.shuffle(training_images)
    training_images = training_images[0:1]
    #training_images=[]
    val_template = r'U:\Data\computer_vision_coursework\face_images\from_both\val\*\*.jpg'
    val_images = glob.glob(val_template)
    #random.shuffle(val_images)
    #val_images = val_images[0:10]
    #train_template = r'C:\Data\computer_vision_coursework\Images\Group11of11\Group11of11\extracted_faces\*.jpg'
    #training_images = glob.glob(train_template)
    #val_images = training_images[0:10]
    this_batch=20

    feature_save_directory=r'.\data\extracted_features_augmented_balanced'
    #feature_save_directory=r'C:\Data\computer_vision_coursework\Images\Group11of11\Group11of11\extracted_faces\feature_data'
    if not os.path.isdir(feature_save_directory):
        os.makedirs(feature_save_directory)

    surf_dict_size=200
    train_bow_surf=None #initialise to None for data flow reasons

    #temp hack: do 2000 files at a time
    tr_all = []
    va_all = []
    sbs=50000
    batches = ceil(len(val_images) / sbs)
    for fbi, fb in enumerate(range(batches)):
        first_tr = fb * sbs
        last_tr = min((1 + fb) * sbs, len(training_images))
        first_va = min(fb * sbs,len(training_images))
        last_va = min((1 + fb) * sbs, len(val_images))

        training_images_to_use = training_images[first_tr:last_tr]
        val_images_to_use = val_images[first_va:last_va]

        results = {}
        for ft in ['surf']:
            print("Start feature extraction for '{}'".format(ft))
            results[ft]={}
            results[ft]['feature_type']=ft

            #Save results in batches to overcome max save size limitations
            save_batch_size = 1000
            if ft!='surf':
                save_nam = 'features_' + ft + '_' + str(len(training_images)) + '_images.npy'
            else:
                #rint('temp')
                #train_bow_surf = get_bow_from_image_list(training_images, dict_size=surf_dict_size)
                train_bow_surf=load_bag_of_words(r'./data/extracted_features/*34744*BOW*npy')
                results[ft]['book_of_words'] = train_bow_surf
                save_nam='features_' + ft + '_dictsize' + str(surf_dict_size) + '_' + str(len(training_images)) + '_images.npy'
                #if surf, save book of words
                rows, cols = np.shape(results[ft]['book_of_words'])
                batches = ceil(rows / save_batch_size)
                for b in range(batches):
                    first = b * save_batch_size
                    last = min((1 + b) * save_batch_size, rows)
                    bow = results[ft]['book_of_words'][first:last]
                    save_nam_batch = save_nam.replace('.npy', '_BOW_batch_{}.npy'.format(str(b)))
                    #np.save(os.path.join(feature_save_directory, save_nam_batch), bow)
                    #print('SURF Book of words batch saved: {}'.format(save_nam_batch))

            #get and store training & validation features
            train_features,train_labels = get_features_for_image_list(
                #training_images,
                training_images_to_use,
                feature_type=ft,
                surf_book_of_words=train_bow_surf
            )
            results[ft]['train_features']=train_features
            results[ft]['train_labels'] = train_labels

            val_features, val_labels = get_features_for_image_list(
                #val_images,
                val_images_to_use,
                feature_type=ft,
                surf_book_of_words=train_bow_surf
            )
            results[ft]['val_features'] = val_features
            results[ft]['val_labels'] = val_labels

            for set in ['val']:
                rows,cols = np.shape(results[ft][set+'_features'])
                batches = ceil(rows/save_batch_size)
                for b in range(batches):
                    first = b*save_batch_size
                    last = min((1+b)*save_batch_size,rows)
                    feats = results[ft][set+'_features'][first:last]
                    results[ft][set + '_features_'+str(b)] = feats
                    labs = results[ft][set+'_labels'][first:last]
                    results[ft][set + '_labels_' + str(b)] = labs
                    results[ft]['set'] = set

                    save_nam_batch = save_nam.replace('.npy','_{}_folderbatch_{}_filebatch_{}_batch_{}.npy'.format(set,this_batch,str(fbi),str(b)))
                    save_keys = ['feature_type', 'set', set + '_features_'+str(b), set + '_labels_' + str(b)]
                    save_dict = {k:results[ft][k] for k in save_keys}
                    np.save(os.path.join(feature_save_directory, save_nam_batch), save_dict)
                    print('Saved feature batch: {}'.format(save_nam_batch))

            results[ft] = None

    print('done!')

