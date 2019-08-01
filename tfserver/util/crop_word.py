import numpy as np
import os
import cv2

from utils.bboxes import polygon_to_rbox

def crop_words(img, boxes, height=32, width=256, grayscale = True):
    
    words = []
    #소영추가
    for j in range(len(boxes)):
        h, w = img.shape[:2]
        
        # polygon case
        box = np.reshape(boxes[j], (-1,2))
        rbox = polygon_to_rbox(box)
        if len(rbox) == 1: 
            continue
        
        word_w, word_h = rbox[2]*w, rbox[3]*h
        
        word_ar = word_w / word_h
        word_h = int(height)
        word_w = int(round(height * word_ar))

        src = np.asarray(box*[w,h], np.float32)
        dst = np.array([
            [0, 0],
            [word_w, 0],
            [word_w, word_h],
            [0, word_h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)

        word = cv2.warpPerspective(img, M, (word_w, word_h), flags=cv2.INTER_CUBIC)
        
        if grayscale:
            word = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
            word = cv2.normalize(word, word, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            word = word[:,:,None]
        
        word = word.astype(np.float32)

        if width is not None:
            tmp_word = word[:,:width,:]
            word = np.ones([height, width, tmp_word.shape[2]])
            word[:,slice(0, tmp_word.shape[1]), :] = tmp_word
        
        words.append(word)
    return words