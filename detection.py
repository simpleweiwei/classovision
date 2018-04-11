import cv2
import imutils
import numpy as np
import glob
import utils as u

def intersect_area(a, b):  # returns None if rectangles don't intersect
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0

def intersection_over_total_area(a,b):

    interArea=intersect_area(a,b)

    boxAArea = (a[2] - a[0] ) * (a[3] - a[1] )
    boxBArea = (b[2] - b[0] ) * (b[3] - b[1] )

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def bb_intersection_over_union(boxA, boxB):
    #https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    #xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    #yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_intersections(bbox_list,overlap_threshold=0.2):
    intersections = []
    for i,bbox in enumerate(bbox_list):
        inter = (i,)
        for j,bbox2 in enumerate(bbox_list):
            if i != j:
                if intersection_over_total_area(bbox, bbox2) > overlap_threshold:
                    inter = inter + (j,)
        intersections.append(inter)
    
    #sort and unique intersections to reduce computation
    intersections = list(set([tuple(sorted(list(x))) for x in intersections]))
        
    return intersections

def get_intersections_pairwise(bbox_list,overlap_threshold=0.2):
    intersections = [(tuple(sorted([x, y])), intersection_over_total_area(xbb, ybb)) for x, xbb in
                     enumerate(bbox_list) for y, ybb in enumerate(bbox_list) if x != y]
    intersections = list(set(intersections))
    # remove intersections below some threshold
    intersections = [x for x in intersections if x[1] > 0.2]

    return intersections

def merge_intersections(bbox_list,overlap_threshold=0.2):

    #find all initial intersections
    intersections = get_intersections(bbox_list, overlap_threshold)

    bbl = bbox_list

    # find max dimensions of any overlapping bboxes and append to list
    merged_bboxes=[]
    all_intersections_ind = []
    for inter in intersections:
        all_intersections_ind = all_intersections_ind + list(inter)
        new_x1 = min([bbl[y][0] for y in inter])
        new_y1 = min([bbl[y][1] for y in inter])
        new_x2 = max([bbl[y][2] for y in inter])
        new_y2 = max([bbl[y][3] for y in inter])
        merged_bboxes.append((new_x1, new_y1, new_x2, new_y2))

        merged_bboxes = list(set(merged_bboxes))

    return merged_bboxes

def try_rotating(text_image):
    #https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
    thresh = cv2.threshold(text_image, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = text_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(text_image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    #print('Trying a rotated version using angle {0}'.format(angle))
    return rotated

def detect_digits(image, debug=False, sharpen=False):
    """
    done with material from https://www.pyimagesearch.com/2017/07/17/credit-card-ocr-with-opencv-and-python/
    :param image:
    :return:
    """
    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    img = imutils.resize(image, width=300)

    if sharpen:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)

    #use global threshold
    th1 = cv2.threshold(img, 165, 255, cv2.THRESH_BINARY_INV)[1]

    # initialize a rectangular (wider than it is tall) and square
    # structuring kernel
    kern = (5, 5)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kern)
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6,6))

    # apply a tophat (whitehat) morphological operator to find light
    # regions against a dark background (i.e., the credit card numbers)
    #tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, rectKernel)
    tophat = cv2.morphologyEx(th1, cv2.MORPH_TOPHAT, rectKernel)

    # compute the Scharr gradient of the tophat image, then scale
    # the rest back into the range [0, 255]
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
                      ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")

    # apply a closing operation using the rectangular kernel to help
    # cloes gaps in between credit card number digits, then apply
    # Otsu's thresholding method to binarize the image
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    #debug: check kernel outputs
    #plt.figure()
    #plt.imshow(thresh)
    #plt.title('using kernel {0}'.format(kern))

    # apply a second closing operation to the binary image, again
    # to help close gaps between credit card number regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    # find contours in the thresholded image, then initialize the
    # list of digit locations
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    locs = []

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # compute the bounding box of the contour, then use the
        # bounding box coordinates to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        cnt_img = img[y:y + h, x:x + w]
        max_colour = np.max(cnt_img)
        min_colour = np.min(cnt_img)

        if debug:
            print('Contour {0} has width {1}, height {2}, ar {3}, max colour {4}, min colour {5}'.format(i,w,h,ar,max_colour,min_colour))
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
            u.imshow(img)
            #imshow(cnt_img)

        # since credit cards used a fixed size fonts with 4 groups
        # of 4 digits, we can prune potential contours based on the
        # aspect ratio
        if (ar > 0.99) & (ar < 2.2) & (max_colour > 185) & (min_colour < 130):
        #if ar > 0:
            # contours can further be pruned on minimum/maximum width
            # and height
            if (w > 8 and w < 45) and (h > 5 and h < 30):
                # append the bounding box region of the digits group
                # to our locations list
                locs.append((x, y, w, h))

    # sort the digit locations from left-to-right, then initialize the
    # list of classified digits
    locs = sorted(locs, key=lambda x: x[0])
    sub_frames = [] # sub_frames is a list of tuples with format [ (i,loc,[digits_for_loc]) ]

    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # initialize the list of group digits

        # extract the group ROI of 4 digits from the grayscale image,
        # then apply thresholding to segment the digits from the
        # background of the credit card
        padd = 5 if (gY > 5) & (gX > 5) else min(gY,gX)
        group = img[gY - padd:gY + gH + padd, gX - padd:gX + gW + padd]

        if 1==0:
            # try sharpen
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            im = cv2.filter2D(group, -1, kernel)

        group = cv2.threshold(group, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # detect the contours of each individual digit in the group,
        # then sort the digit contours from left to right
        group2 = 255-group
        group2 = try_rotating(group2)
        group2 = imutils.resize(group2,width=28*2+padd*2,height=28+padd*2)

        digitCnts = cv2.findContours(group2, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]
        #digitCnts = contours.sort_contours(digitCnts,
        #                                   method="left-to-right")[0]

        # loop over the digit contours
        digits=[]
        digit_locs=[]
        for c in digitCnts:
            # compute the bounding box of the individual digit, extract
            # the digit, and resize it to have the same fixed size as
            # the reference OCR-A images
            (x, y, w, h) = cv2.boundingRect(c)

            mask = np.zeros(group2.shape, np.uint8)
            m = cv2.drawContours(mask, [c], 0, 255, -1)

            blankmat = np.zeros(group2.shape, np.uint8)
            idx = (mask != 0)
            blankmat[idx] = group2[idx]

            padd2 = 1 if (y < 1) & (x < 1) else min(y, x)
            roi = blankmat[y - padd2:y + h + padd2, x - padd2:x + w + padd2]
            # check that 'digit' width is not more than 70% of double number width
            if np.shape(roi)[1] < np.shape(group2)[1]*0.7:
                if min(np.shape(roi)) > 0:
                    roi = cv2.resize(roi, (28, 28))
                    digits.append(roi)
                    digit_locs.append((x,y,w,h))

            img = cv2.rectangle(img, (gX + x - padd, gY + y - padd), (gX + x + w - padd, gY + y + h - padd), (0,0,255), 1)

        #enforce ordering of digit lists by x coordinate to get order correct
        digit_sort_order = np.argsort([x[0] for x in digit_locs])
        digits = [digits[j] for j in digit_sort_order]
        digit_locs = [digit_locs[j] for j in digit_sort_order]

        sub_frame = (i,group2,digits)
        sub_frames.append(sub_frame)

    return (locs,img,sub_frames)

def detect_faces(image, model_file, prototxt_file, min_confidence, aspect_ratio_bounds=(0.8,1.4), merge_overlap=0.2, stepSize=700, window_size=(1000,1000)):
    #https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(prototxt_file, model_file)

    #Run face detections in windows across image
    #stepSize=700
    #window_size=(1000,1000)
    window_list = []
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            window_list.append(((y,x),image[y:y + window_size[1], x:x + window_size[0],:]))

    all_bboxes = []
    for window in window_list:
        y,x=window[0]
        img = window[1]
        (h, w) = img.shape[:2]
        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > min_confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # check aspect ratios within certain bounds for faces
                ar = (endY - startY) / (endX - startX)
                #print('Aspect ratio: {}'.format(ar))
                #u.imshow(image[y+startY:y+endY, x+startX:x+endX])
                if (ar > aspect_ratio_bounds[0]) & (ar < aspect_ratio_bounds[1]):
                    all_bboxes.append(((x+startX, y+startY, x+endX, y+endY),confidence))

    #find and merge overlapping bboxes
    all_bboxes0 = [x[0] for x in all_bboxes]
    if merge_overlap is not None:
        all_bboxes0 = merge_intersections(all_bboxes0, merge_overlap)
    return all_bboxes0


if __name__ == '__main__':


    print('Done!')

