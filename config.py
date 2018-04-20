CASCADE_FOLDER=r"C:\GitRepo\opencv\data\haarcascades"
HAAR_POSITIVE_FOLDER=r"C:\Data\computer_vision_coursework\Images\haar_images\positive"
HAAR_NEGATIVE_FULLIMAGE_FOLDER=r"C:\Data\computer_vision_coursework\Images\haar_images\negative"
HAAR_NEGATIVE_SUBIMAGE_FOLDER=r"C:\Data\computer_vision_coursework\Images\haar_images\negative\sub_images"

frame_frac=0.2
unknown_face_return_value=9999

digit_cnn_path=r'.\saved_networks\cnn_alldignan.h5'

# models paths
face_cnn = r'saved_networks/vgg_face61_epoch4.h5'
svm_hog=r'saved_models/hog_svm_trained_48800obs.pck'
svm_surf=r'saved_models/surf_svm_trained_48800obs.pck'
svm_lbp=r'saved_models/lbp_svm_trained_48800obs.pck'
mlp_hog=r'saved_models/hog_mlp_trained_48800obs.pck'
mlp_surf=r'saved_models/surf_mlp_trained_48800obs.pck'
mlp_lbp=r'saved_models/lbp_mlp_trained_48800obs.pck'
rf_hog=r'saved_models/hog_rf_trained_48800obs.pck'
rf_surf=r'saved_models/surf_rf_trained_48800obs.pck'
rf_lbp=r'saved_models/lbp_rf_trained_48800obs.pck'
nb_hog=r'saved_models/hog_nb_trained_48800obs.pck'
nb_surf=r'saved_models/surf_nb_trained_48800obs.pck'
nb_lbp=r'saved_models/lbp_nb_trained_48800obs.pck'

# saved book of words for surf vocab hist gen
bow_file=r'bag_of_words_BOW.npy'

# SSD face detector model files
ssd_model=r'saved_networks\res10_300x300_ssd_iter_140000.caffemodel'
prototxt_file=r'saved_networks\deploy.prototxt.txt'

# face CNN class labels
class_labels=['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014',
              '015', '016', '017', '018', '035', '037', '045', '046', '050', '051', '052', '053', '054', '055',
              '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069',
              '070', '107', '108', '161', '164', '165', '166', '167', '168', '169', '170', 'unknown_1', 'unknown_2',
              'unknown_3', 'unknown_4', 'unknown_5', 'unknown_6', 'unknown_7', 'unknown_8']
