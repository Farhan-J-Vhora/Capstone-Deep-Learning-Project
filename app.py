import numpy as np
import cv2 as cv

from sklearn.metrics import pairwise

background = None

accumulated_weight = 0.5

roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600

def calc_accum_avg(frame, accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype('float')
        return None
    
    cv2.accumulateWeighted(frame,background,accumulated_weight)

def segment(frame, threshold_min=25):

    diff = cv2.absdiff(background.astype('unit8'), frame)

    ret, thresholded = cv2.threshold(diff, threshold_min, 255, cv2.THRESH_BINARY)

    image, contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresholded,hand_segment)
    
def count_fingers(thresholded, hand_segment):

    conv_hull = cv2.convexHull(hand_segment)

    top = tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:,:,0].argmax()][0])

    cX = (left[0]+right[0])//2
    cY = (top[0]+bottom[1])//2

    distance = pairwise.euclidean_distances([cX,cY], Y=[left,right,top,bottom])[0]

    max_distance = distance.max()

    radius = int(0.9*max_distance)

    circumfrence = (2*np.pi*radius)

    circular_roi = np.zeroes(thresholded[:2],dtype='unit8')

    cv2.circle(circular_roi,(cX,cY),radius,255,10)

    circular_roi = cv2.bitwise_and(thresholded,thresholded,mask=circular_roi)

    image, contours, hierarchy = cv2.findContours(circular_roi.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count = 0

    for cnt in contours:

        (x,y,w,h) = cv2.boundingRect(cnt)
        out_of_wrist = (cY + (cY * 0.25)) > (y+h)

        limit_points = ((circumfrence*0.25) > cnt.shape[0])

        if out_of_wrist and limit_points:
            count += 1
    
    return count