import cv2
import glob
from sklearn import metrics
from keras.models import load_model

from utils import imshow, get_class_labels_from_folders

from classification import classify_individual_digit


if __name__ == '__main__':
    """
    directory_list = [
        r'C:\Data\computer_vision_coursework\Images\digit_images\ind_digits\0',
        r'C:\Data\computer_vision_coursework\Images\digit_images\ind_digits\1',
        r'C:\Data\computer_vision_coursework\Images\digit_images\ind_digits\2',
        r'C:\Data\computer_vision_coursework\Images\digit_images\ind_digits\3',
        r'C:\Data\computer_vision_coursework\Images\digit_images\ind_digits\4',
        r'C:\Data\computer_vision_coursework\Images\digit_images\ind_digits\5',
        r'C:\Data\computer_vision_coursework\Images\digit_images\ind_digits\6',
        r'C:\Data\computer_vision_coursework\Images\digit_images\ind_digits\7',
        r'C:\Data\computer_vision_coursework\Images\digit_images\ind_digits\8',
        r'C:\Data\computer_vision_coursework\Images\digit_images\ind_digits\9',
        r'C:\Data\computer_vision_coursework\Images\digit_images\ind_digits\negatives'
    ]

    aggregate_and_save_training_data(directory_list)    
    """

    folder = r'C:\Data\computer_vision_coursework\Images\digit_images\ind_digits\*\*jpg'
    files = glob.glob(folder)
    print('Running digit classification for {} digit files'.format(len(files)))
    labels = get_class_labels_from_folders(files)

    model1 = load_model(r'saved_networks\mnist_cnn.h5')
    model2 = load_model(r'saved_networks\cnn_alldignan.h5')

    model1_pred = []
    model2_pred = []
    for i,file in enumerate(files):
        #print('Running for image {} of {}'.format(i,len(files)))
        img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        model1_pred.append(str(classify_individual_digit(model1, img)))
        model2_pred.append(str(classify_individual_digit(model2, img)))

    model1_pred = [x.replace('10', 'negatives') for x in model1_pred]
    model2_pred = [x.replace('10','negatives') for x in model2_pred]

    label_order = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'negatives']

    cm1 = metrics.confusion_matrix(labels, model1_pred, labels=label_order)
    cm2 = metrics.confusion_matrix(labels, model2_pred, labels=label_order)


    print('Done!')
