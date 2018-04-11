import cv2
import numpy as np
import os
import glob
import random
import imutils
import shutil
import matplotlib.pyplot as plt
from sklearn import metrics
import utils as u
from detection import detect_faces, detect_digits
from classification import classify_individual_digit, identify_digit_string, classify_face, get_face_cnn


def main():

    #testing classifiers on group images
    test_labels = [os.path.basename(x) for x in glob.glob(r'C:\Data\computer_vision_coursework\Images\individual_people\val\*')]
    test_labels = test_labels + ['unknown_1', 'unknown_2', 'unknown_3', 'unknown_4', 'unknown_5', 'unknown_6']
    test_folder = r"C:\Data\computer_vision_coursework\Images\individual_people\val\*\*jpg"
    test_img_list = glob.glob(test_folder)
    random.shuffle(test_img_list)
    #test_img_list=test_img_list[0:n_test]

    model_file=r'saved_networks\res10_300x300_ssd_iter_140000.caffemodel'
    prototxt_file=r'saved_networks\deploy.prototxt.txt'
    merge_overlap=0.6
    aspect_ratio_bounds = (0.8, 1.4)
    min_confidence=0.6
    step_size=250
    window_size=(500,500)

    results = {}

    for model in ['cnn','nb','svm','mlp','rf',]:
        results[model] = {}
        for feat_type in ['hog','surf','lbp',None]:
            if (feat_type is None) & (model != 'cnn'):
                continue
            if (model == 'cnn') & (feat_type is not None):
                continue
            print("Begin validation testing on model '{}', feature type '{}'".format(model,feat_type))
            y = []
            t = []
            for test_img in test_img_list:
                image = cv2.imread(test_img)
                face_bboxes = detect_faces(image, model_file, prototxt_file, min_confidence,aspect_ratio_bounds, merge_overlap, step_size, window_size)

                for face_bbox in face_bboxes:
                    x1,y1,x2,y2=face_bbox
                    face_img = image[y1:y2,x1:x2]
                    correct_label = u.get_class_label_from_folder(test_img)
                    try:
                        face_id = classify_face(face_img,method=model,feature_type=feat_type)
                        if face_id is not None:
                            if type(face_id) == np.int64:
                                face_label = test_labels[face_id]
                            else:
                                face_label = face_id[0]
                            y.append(face_label)
                            t.append(correct_label)
                            #print('Prediction: {}, Actual: {}, Status: {}'.format(face_label,correct_label,face_label==correct_label))
                    except Exception as ex:
                        print("Error on model '{}' feature type '{}' image '{}' \n '{}'".format(model,feat_type,test_img,ex))

            try:
                results[model][feat_type] = metrics.confusion_matrix(t,y,labels=test_labels)
            except Exception as ex:
                print('Error printing results to dict: {}'.format(ex))
                results[model][feat_type] = ex
            np.save('accuracy_results.npy', results)

        # bbox_classifications.append((face_bbox,face_id))
        # loc = (int(round(np.mean([x1,x1]))),int(round(np.mean([y1,y2]))))
        # print('Found face {} at location {}'.format(face_id,loc))
        # cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # #cv2.putText(img2, str(face_id), loc, cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)



    # identifying digits
    glob_template = r"C:\Data\computer_vision_coursework\Images\individual_people\061\IMG_0861.jpg"
    img_files = glob.glob(glob_template)
    #random.shuffle(img_files)
    #img_files = img_files[0:10]

    model_path=r'saved_networks\cnn_alldignan.h5'
    #digit_cnn = load_model(r'saved_networks\cnn_alldignan.h5')

    class_labels=[]
    class_pred=[]
    for i,file in enumerate(img_files):
        print('Checking file {} of {}: {}'.format(i+1,len(img_files),file))
        image = cv2.imread(file)
        class_labels.append(u.get_class_label_from_folder(file)[1:3])
        class_pred.append(identify_digit_string(image, model_path))

    cm = metrics.confusion_matrix(class_labels,class_pred)



    # classifying and moving extracted faces from group photos
    read_glob=r'C:\Data\computer_vision_coursework\Images\Group11of11\Group11of11\extracted_faces\clustering_2\*jpg'
    input_files = glob.glob(read_glob)

    label_folders=r'C:\Data\computer_vision_coursework\Images\face_images\train\*'
    labels = [os.path.basename(x) for x in glob.glob(label_folders)]

    for file in input_files:
        face_img = cv2.imread(file)
        if any([x == 0 for x in np.shape(face_img)]):
            continue
        face_id = classify_face(face_img, method='cnn', feature_type=None)
        to_dir = os.path.join(os.path.dirname(file),'classified',labels[face_id])
        if not os.path.isdir(to_dir):
            os.mkdir(to_dir)
        to_file = os.path.join(to_dir,os.path.basename(file))
        shutil.copy(file,to_file)

    #extracting faces from group photos
    write_path = r"C:\Data\computer_vision_coursework\Images\Group11of11\Group11of11\extracted_faces"
    group_photo_glob = r"C:\Data\computer_vision_coursework\Images\Group11of11\Group11of11\*.jpg"
    gf_files = glob.glob(group_photo_glob)

    i = 0
    for gf in gf_files:
        image = cv2.imread(gf)
        faces_bboxes = face_bboxes = detect_faces(image, model_file, prototxt_file, min_confidence, aspect_ratio_bounds, merge_overlap,
                               step_size, window_size)
        print('{} faces found in group photo'.format(np.shape(faces_bboxes)[0]))
        for face_bbox in faces_bboxes:
            x1,y1,x2,y2=face_bbox
            face_img = image[y1:y2,x1:x2]
            if not all([x > 0 for x in np.shape(face_img)]):
                continue
            face_img = cv2.resize(face_img,(224,224))
            write_file = os.path.join(write_path,'group_face_{}.jpg'.format(i))
            cv2.imwrite(write_file,face_img)
            print('Face image written: {}'.format(write_file))
            i+=1



    return

if __name__ == '__main__':

    main()
    print('Done!')