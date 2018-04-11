import glob
import os
import random
from math import floor

import cv2
import imutils
import numpy as np

import config as cfg


class ObjectDetector(object):
    def detect_features(self, image, cascades={}, draw_rectangle=True, window_name='img'):
        image = image.copy()
        for c in cascades.keys():
            cf = cascades[c]['file_name']
            assert os.path.isfile(cf), "Cascade file {0} doe not exist".format(cf)

            cascades[c]['cascade'] = cv2.CascadeClassifier(cf)
            cascades[c]['results'] = cascades[c]['cascade'].detectMultiScale(image, 1.3, 5)

            if draw_rectangle:
                o = cascades[c]['offset']
                for (x, y, w, h) in cascades[c]['results']:
                    img = cv2.rectangle(image, (x - o, y - o), (x + w + o, y + h + o), cascades[c]['line_colour'], 2)
            else:
                img = image

        return (cascades, img)

    def detect_eyes(self, image, draw_rectangle=True, window_name='img'):

        cascades = {
            'eye_1': {
                'file_name': os.path.join(cfg.CASCADE_FOLDER, 'haarcascade_eye.xml'),
                'offset': 0,
                'line_colour': (255, 0, 0)
            },
            'eye_2': {
                'file_name': os.path.join(cfg.CASCADE_FOLDER, 'haarcascade_eye_tree_eyeglasses.xml'),
                'offset': 3,
                'line_colour': (0, 255, 0)
            },
            'eye_3': {
                'file_name': os.path.join(cfg.CASCADE_FOLDER, 'haarcascade_lefteye_2splits.xml'),
                'offset': 6,
                'line_colour': (0, 0, 255)
            },
            'eye_4': {
                'file_name': os.path.join(cfg.CASCADE_FOLDER, 'haarcascade_righteye_2splits.xml'),
                'offset': 9,
                'line_colour': (125, 0, 125)
            },
        }

        return self.detect_features(image, cascades, draw_rectangle)

    def detect_faces(self, image, draw_rectangle=True, window_name='img'):
        cascades = {
            'face_1': {
                'file_name': os.path.join(cfg.CASCADE_FOLDER, 'haarcascade_frontalface_alt.xml'),
                'offset': 1,
                'line_colour': (255, 0, 0)
            },
            'face_2': {
                'file_name': os.path.join(cfg.CASCADE_FOLDER, 'haarcascade_frontalface_alt_tree.xml'),
                'offset': 1,
                'line_colour': (0, 255, 0)
            },
            'face_3': {
                'file_name': os.path.join(cfg.CASCADE_FOLDER, 'haarcascade_frontalface_alt2.xml'),
                'offset': 1,
                'line_colour': (0, 0, 255)
            },
            'face_4': {
                'file_name': os.path.join(cfg.CASCADE_FOLDER, 'haarcascade_frontalface_default.xml'),
                'offset': 1,
                'line_colour': (125, 0, 125)
            }
            #,
            #'sarah_face': {
            #    'file_name': os.path.join(cfg.CASCADE_FOLDER, 'sarahFaceDetector_haar.xml'),
            #    'offset': 12,
            #    'line_colour': (0, 125, 125)
            #}
        }

        return self.detect_features(image, cascades, draw_rectangle)


def get_faces_from_ind_photos_using_viola_jones():
    glob_template = r"C:\Data\computer_vision_coursework\Images\individual_people\*"
    img_folders = glob.glob(glob_template)
    img_folders = [x for x in img_folders if ('digit_images' not in x) and ('MACOSX' not in x)]
    img_folders = [x for x in img_folders if ('11of11' not in x)]
    img_folders = [x for x in img_folders if len(os.path.basename(x))==3]
    for folder in img_folders:
        print('Running through folder {}'.format(folder))
        faces_dir = os.path.join(folder,r'extracted_faces')
        if not os.path.isdir(faces_dir):
            os.mkdir(faces_dir)

        local_glob_template = os.path.join(folder,'*jpg')
        img_files = glob.glob(local_glob_template)
        random.shuffle(img_files)
        img_files = img_files[0:2]
        if len(img_files) > 0:
            pos_write_folder = r'C:\Data\computer_vision_coursework\Images\haar_images\positive_individual_faces'
            neg_write_folder = r'C:\Data\computer_vision_coursework\Images\haar_images\negative_body'

            od = ObjectDetector()
            face_id = range(1)

            for i,file in enumerate(img_files):
                print('Checking file {}'.format(file))
                all_faces_bbox = []
                fn = os.path.basename(file)
                image = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
                (face_cascades, img) = od.detect_faces(image,draw_rectangle=False)
                # take list of all detected faces from existing cascades
                for cascade in face_cascades.keys():
                    results = face_cascades[cascade]['results']
                    for r in range(np.shape(results)[0]):
                        all_faces_bbox.append(results[r, :])


                for f,face_bbox in enumerate(all_faces_bbox):
                    [x,y,w,h] = face_bbox
                    face_img = img[y+1:y + h -1, x+1:x + w -1]
                    #ffn = os.path.basename(file).replace('.jpg', '_face_{}_{}.jpg'.format(f,sum(sum(face_img))))
                    ffn = 'face_{}_{}_{}'.format(sum(sum(face_img)),f,os.path.basename(file))
                    pos_save_path = os.path.join(pos_write_folder,ffn)
                    if np.shape(face_img)[0:2] != (0,0):
                        face_img = imutils.resize(face_img,width=50,height=50)
                        cv2.imwrite(pos_save_path,face_img)

                image2 = blank_out_ranges(image,all_faces_bbox)
                neg_save_path = os.path.join(neg_write_folder,os.path.basename(file))
                cv2.imwrite(neg_save_path,image2)

    return


def manual_get_faces_for_haar():
    img_files = [
        r"C:\Data\computer_vision_coursework\Images\11of11\11of11\180\IMG_0613.jpg",
        r"C:\Data\computer_vision_coursework\Images\11of11\11of11\180\IMG_0651.jpg",
        r"C:\Data\computer_vision_coursework\Images\11of11\11of11\180\IMG_0627_62.jpg",
        r"C:\Data\computer_vision_coursework\Images\11of11\11of11\180\IMG_0627.jpg",
        r"C:\Data\computer_vision_coursework\Images\11of11\11of11\180\IMG_0626.jpg"
    ]
    glob_template = r"C:\Data\computer_vision_coursework\Images\11of11\11of11\180\*.jpg"
    #glob_template = r"C:\Data\computer_vision_coursework\Images\*\*\*\*jpg"
    img_files = glob.glob(glob_template)
    random.shuffle(img_files)
    img_files = img_files[0:100]

    #pos_write_folder = './data/haar_face_positives'
    #neg_write_folder = './data/haar_face_negatives'
    pos_write_folder = cfg.HAAR_POSITIVE_FOLDER
    neg_write_folder = cfg.HAAR_NEGATIVE_FULLIMAGE_FOLDER

    od = ObjectDetector()
    face_id = range(1)

    for i,file in enumerate(img_files):
        print('Checking file {}'.format(file))
        all_faces_bbox = []
        fn = os.path.basename(file)
        image = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        (face_cascades, img) = od.detect_faces(image)
        # take list of all detected faces from existing cascades
        for cascade in face_cascades.keys():
            results = face_cascades[cascade]['results']
            for r in range(np.shape(results)[0]):
                all_faces_bbox.append(results[r, :])

        #img2 = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
        face_roi = {}
        fromCenter = False
        for f in face_id:
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            face_bbox = cv2.selectROI('Image',img, False, False)
            if face_bbox != (0,0,0,0):
                all_faces_bbox.append(np.array(face_bbox))
            (x,y,w,h) = face_bbox
            face = img[y:y + h,x:x + w]
            face_roi[f] = face
            ffn = fn.replace('.jpg','_{}.jpg'.format(f))
            pos_save_path = os.path.join(pos_write_folder,ffn)
            if np.shape(face)[0:2] != (0,0):
                face = imutils.resize(face,width=50,height=50)
                cv2.imwrite(pos_save_path,face)

        image2 = blank_out_ranges(image,all_faces_bbox)
        neg_save_path = os.path.join(neg_write_folder,os.path.basename(file))
        cv2.imwrite(neg_save_path,image2)

    return


def slice_up_image(image,slice_size=(30,30)):
    im_size = np.shape(image)
    slices = []
    for x_window in range(floor(im_size[0]/slice_size[0])):
        for y_window in range(floor(im_size[1] / slice_size[1])):
            x = slice_size[0]*x_window
            y = slice_size[1]*x_window
            w = slice_size[0]
            h = slice_size[1]
            #print('Window: ({}:{},{}:{})'.format(x,x+w,y,y+h))
            slices.append(image[y:y+h,x:x+w])

    return slices


def slice_and_save_images(glob_path,save_dir,window_size=(100,100)):
    img_paths = glob.glob(glob_path)
    for file in img_paths:
        filename = os.path.basename(file)
        img = cv2.imread(file)
        img_slices = slice_up_image(img,window_size)
        for i,slice in enumerate(img_slices):
            save_name = filename.replace('.jpg','_window{}.jpg'.format(i))
            save_path = os.path.join(save_dir,save_name)
            cv2.imwrite(save_path,slice)


def blank_out_ranges(image,list_of_bbox):
    img = image.copy()
    shp = np.shape(img)
    if len(shp) == 2:
        #add channels dimension if necessary
        img = np.reshape(img, shp + (1,))

    for rng in list_of_bbox:
        (x,y,w,h) = rng
        img[y:y+h,x:x+w,:] = 0

    return img