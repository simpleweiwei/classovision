import datetime as dt
import glob
import os
import sys
import imutils
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import random
from math import ceil,floor
import shutil

from classification import get_digit_class_dict



def is_module_imported(module_name):
    return True if module_name in sys.modules else False


def imshow(image,label='img'):
    cv2.imshow(label, image)
    cv2.waitKey(0)
    cv2.destroyWindow(label)


def get_formatted_data(file,output_dim=(1,28,28,1), dir_to_class_ind={}):

    img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    x = img.reshape(*output_dim)

    ind = get_class_label_from_folder(file,dir_to_class_ind)
    y = np.zeros(len(dir_to_class_ind))
    y[ind] = 1
    y = y.reshape(1,len(dir_to_class_ind))

    return (x,y)


def save_nd_array(arr,path):
    arr = arr.reshape(np.prod(np.shape(arr)))
    np.save(path,arr)


def save_2d_array(arr,path):
    np.save(path, arr)


def load_nd_array(path,dims=None):
    #override 1st dim to -1 to allow automatic setting
    dims = (-1,*dims[1:4])
    arr = np.load(path)
    return arr.reshape(dims)


def load_2d_array(path):
    return np.load(path)


def aggregate_and_save_training_data(directory_list):
    """
    Function reads a list of folders and produces an array with x and y data for all images
    images are labelled by folder name (eg 6,7, nan...)
    :param directory_list:
    :return:
    """
    files = []

    for base_path in directory_list:
        glob_template=os.path.join(base_path,r"*.jpg")
        files = files + glob.glob(glob_template)

    x_data = None
    y_data = None
    x_output_dim=(1,28,28,1)
    cd = get_digit_class_dict()

    for i,file in enumerate(files):
        x,y = get_formatted_data(file, x_output_dim, dir_to_class_ind=cd)

        x_data = x if x_data is None else np.concatenate([x_data,x], axis=0)
        y_data = y if y_data is None else np.concatenate([y_data, y], axis=0)
        print('Read file {} out of {}: {}'.format(i,len(files),file))

    save_path = r'./data/{}_data_{}.npy'.format('{}',dt.datetime.now().strftime('%Y%m%d%H%m'))
    save_nd_array(x_data, save_path.format('x'))
    save_2d_array(y_data, save_path.format('y'))
    print('Written x data with dims {} and y data with dims {}'.format(np.shape(x_data),np.shape(y_data)))


def get_save_video_frames(mov_file,rotate=False):

    vidcap = cv2.VideoCapture(mov_file)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if image is not None:
            if rotate:
                image = np.transpose(image, (1, 0, 2))
                image = np.flip(image,1)
            output_path = mov_file.replace('.mov',"_%d.jpg" % count)
            cv2.imwrite(output_path, image)  # save frame as JPEG file
            count += 1
    return count

def try_watershed(image):
    #https://www.pyimagesearch.com/2015/11/02/watershed-opencv/

    image = image/255
    image = cv2.resize(image, (0,0), fy=2, fx=2)

    kernel = np.ones((4, 4), np.uint8)
    thresh = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    thresh2 = cv2.erode(image, kernel, iterations=1)

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=5,labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(image.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

    return

def rescale_all_images(glob_pattern,output_size=(50,50)):
    img_list = glob.glob(glob_pattern)

    for file in img_list:
        print('Reading image {}'.format(file))
        img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        img = imutils.resize(img,width=output_size[0],height=output_size[1])
        cv2.imwrite(file,img)
        print('Written {}'.format(file))

def open_save_grayscale(glob_pattern):
    img_list = glob.glob(glob_pattern)

    for file in img_list:
        print('Reading image {}'.format(file))
        img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(file,img)
        print('Written {}'.format(file))

def get_class_label_from_folder(image_file):
    return os.path.basename(os.path.dirname(image_file))

def get_class_labels_from_folders(image_files):
    class_labels=[]
    for img_file in image_files:
        class_label = get_class_label_from_folder(img_file)
        class_labels.append(class_label)
    return class_labels

def move_from_train_to_val(train_folder,val_folder,train_frac):
    #val_folder = r'C:\Data\computer_vision_coursework\Images\face_images\validation'
    #train_folder = r'C:\Data\computer_vision_coursework\Images\face_images\train'
    for folder in glob.glob(os.path.join(train_folder,'*')):
        image_files = glob.glob(os.path.join(folder,'*.jpg'))
        val_count = ceil(len(image_files)*train_frac)
        random.shuffle(image_files)
        val_files = image_files[0:val_count]
        to_folder=os.path.join(val_folder,os.path.basename(folder))
        if not os.path.isdir(to_folder):
            os.mkdir(to_folder)

        for valf in val_files:
            to_file=os.path.join(to_folder,os.path.basename(valf))
            shutil.move(valf,to_file)

def is_error_image(image_file):
    try:
        image = cv2.imread(image_file)
    except:
        return True

    if image is None:
        return True
    else:
        return False



