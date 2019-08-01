from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import HttpResponse
from django.http import JsonResponse

import argparse
import json
import numpy as np
import requests
import cv2
import os
import sys

from PIL import Image
from PIL.ExifTags import TAGS

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, connections

from tfserving.utils import sl_util, crop_word
from tfserving.utils.crnn_utils import decode
from tfserving.utils.crnn_utils import dict838 as dic
from tfserving.utils.bboxes import rbox_to_polygon

from io import BytesIO
import base64
import matplotlib.pyplot as plt



@csrf_exempt
def predict_image(request):
    #print('image load start')
    img ,img_info, points = load_image_from_android(request)
    #print('image load end')
    gps = get_gps_from_image(img_info)
    #print('tfserving start')
    predicted_shop_name = tfserving_img(img, points, segment_threshold = 0.4, link_threshold = 0.25) # return list of text -> just one
    
    if len(predicted_shop_name) == 0:
        predicted_shop_name = ' '
    else:
        predicted_shop_name = predicted_shop_name[0]

    print(predicted_shop_name)
    #print('tfserving end')

    #print('elastic start')
    shop_name, response_es = send_to_elasticsearch(predicted_shop_name,gps)
    print(shop_name)
    #print('elastic end')
    
    #url_list = get_images_from_file_server(shop_name)
    
    url_list = ['0.jpg','1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg']
    
    response_to_android = {'url_list':url_list, 'response_es': response_es}

    response_for_android = JsonResponse(response_to_android, json_dumps_params={'ensure_ascii':False})
    print('end!')
    return response_for_android

@csrf_exempt
def load_image_from_android(request):
    uploadedFile = request.FILES['file'] # django.core.files.uploadedfile.TemporaryUploadedFile
    img_bytes = uploadedFile.read() # bytes
    
    #print("image load")

    img = Image.open(BytesIO(img_bytes))
    
    img_np = np.asarray(img, dtype=np.float32)
    img_info = img.getexif()

    points = request.POST['touchPoint']
    #print("points load")

    return img_np, img_info, points

def get_images_from_file_server(shop_name):
    
    url_list = []

    real_thumbnail_path= 'www_dir/static/'+shop_name+'/thumbnail/'
    #real_thumbnail_path= '/www_dir/static/'+shop_name+'/thumbnail/'
    for filename in os.listdir(real_thumbnail_path):
        
        url_list.append(filename)

    real_menu_path= 'www_dir/static/'+shop_name+'/menu/'
    #real_menu_path= '/www_dir/static/'+shop_name+'/menu/'
    for filename in os.listdir(real_menu_path):
        url_list.append(filename)

    print(url_list)

    return url_list

def get_gps_from_image(img_info):

    exif = {}
    for tag, value in img_info.items():
        decoded = TAGS.get(tag, tag) 
        exif[decoded] = value
    
    exifGPS = exif['GPSInfo']

    try:
        latData = exifGPS[2]
        lonData = exifGPS[4]
        # calculae the lat / long
        latDeg = latData[0][0] / float(latData[0][1]) 
        latMin = latData[1][0] / float(latData[1][1])
        latSec = latData[2][0] / float(latData[2][1])
        lonDeg = lonData[0][0] / float(lonData[0][1])
        lonMin = lonData[1][0] / float(lonData[1][1])
        lonSec = lonData[2][0] / float(lonData[2][1])
        # correct the lat/lon based on N/E/W/S
        Lat = (latDeg + (latMin + latSec / 60.0) / 60.0)
        if exifGPS[1] == 'S': Lat = Lat * -1
        Lon = (lonDeg + (lonMin + lonSec / 60.0) / 60.0)
        if exifGPS[3] == 'W': Lon = Lon * -1
        
        gps = {"Lat":Lat, "Lon":Lon}
    except:
        gps = {"Lat":0, "Lon":0}

    return gps

def send_to_elasticsearch(shop_name,gps_info):

    Lat = gps_info['Lat']
    Lon = gps_info['Lon']


    client = Elasticsearch('http://127.0.0.1:9202')

    a=0
    for i in shop_name:
        if a==0:
            b='{"wildcard" : {"상호명": "'+i+'*"}},'
            a=a+1
        elif a==len(shop_name)-1:
            b=b+'{"wildcard" : {"상호명": "*'+i+'"}},'
            a=a+1
        else:
            b=b+'{"wildcard" : {"상호명": "*'+i+'*"}},'
            a=a+1
    if Lat!=0 & Lon!=0:
        body_sample='{\
                "query": {\
                "bool" : {\
                    "should" : [\
                        '+b+'\
                        { "match": {\
                "상권업종대분류명": "음식"\
                }}\
                    ],\
                    "filter" : {\
                        "geo_bounding_box" : {\
                            "위치": {\
                                "top_left" : {\
                                    "lat" : '+str(Lat+0.05)+',\
                                    "lon" : '+str(Lon-0.05)+'\
                                },\
                                "bottom_right" : {\
                                    "lat" : '+str(Lat-0.05)+',\
                                    "lon" : '+str(Lon+0.05)+'\
                                }\
                            }\
                        }\
                    }\
                }\
            }\
        }'
    else:
        body_sample='{\
            "query": {\
            "bool" : {\
                "should" : [\
                    '+b+'\
                ]\
            }\
        }\
    }'

    body_sample2='{\
        "query": {\
        "bool" : {\
            "should" : [\
                '+b+'\
            {"match_all":{\
            }}\
            ]\
        }\
    }\
    }'

    data2 =client.search(index='storedocu', body=body_sample2)
    
    response_es =data2['hits']['hits'][0]['_source']

    shop_name = response_es['상호명']

    return shop_name, response_es

def tfserving_img(input_image,points, segment_threshold = 0.45, link_threshold = 0.3):

    print(input_image.shape)

    # cv2.INTER_LINEAR, cv2.INTER_CUBIC
    input_seglink = cv2.resize(input_image, (512,512), cv2.INTER_AREA) # (512,512,3)
    
    ax, ay = points.split(',')
    
    image_x = float(ay)*512
    image_y = 512-float(ax)*512
    point_array = np.array([image_x,image_y],dtype=np.float32)

    input_seglink2 = input_seglink.astype('int32')
        
    payload_seglink = {
        "instances": [{'input_1:0': input_seglink.tolist()}]
    }

    print('seglink predict start')
    res_seglink = requests.post('http://tfserving-adress-replace-this-part/v1/models/seglink:predict', json=payload_seglink)
    print('seglink end')
    pred_seglink = json.loads(res_seglink.content.decode('utf-8'))
    pred_seglink_np = np.array(pred_seglink['predictions']) # (1, 5461, 31)
    pred_seglink_np = np.squeeze(pred_seglink_np, axis=0) # (5461, 31)

    prior_util = sl_util.PriorUtil()
    
    boxes = prior_util.decode(pred_seglink_np, segment_threshold, link_threshold, debug=False)
    
    xylist = []
    xy_raw = []
    dist_list = []
    for box in range(len(boxes)):
        cx, cy = boxes[box][:2]
        center_array = np.array([cx,cy], dtype=np.float32)
        dist = np.linalg.norm(center_array-point_array)
        
        xy_rec = rbox_to_polygon(boxes[box][:5])    
        xy_rec = np.flip(xy_rec, axis = 0)
        xy_raw.append(xy_rec)
        xy_r = [r/512 for r in xy_rec]
        xylist.append(xy_r)
        dist_list.append(dist)

    print(len(xylist),"개 박스 발견")

    if len(xylist) !=0:
        min_dist_index = dist_list.index(min(dist_list))
        xylist_new = [xylist[min_dist_index]]
    

    words, vac = crop_word.crop_words(input_seglink, xylist_new, grayscale=True)

    xy_raw = np.delete(xy_raw, vac,0)
    
    word_images = np.ones([len(words), 256, 32, 1])
    for j in range(len(words)):
        word_images[j] = words[j].transpose(1,0,2)
        
    '''
    for i in range(len(word_images)):
        cv2.imshow('boxes'+str(i),np.array(word_images[i][:,:,0], dtype=np.uint8))
    '''
    
    print('crnn predict start')
    text_list = []
    for input_crnn in word_images:

        payload_crnn = {
            "instances": [{'image_input:0' : input_crnn.tolist()}]
        }
        
        res_crnn = requests.post('http://tfserving-adress-replace-this-part/v1/models/crnn:predict', json=payload_crnn)

        pred_crnn = json.loads(res_crnn.content.decode('utf-8'))
        pred_crnn_np = np.array(pred_crnn['predictions']) # (1, 62, 836)
        pred_crnn_np = np.squeeze(pred_crnn_np, axis=0) # (62, 836)

        chars = [dic[c] for c in np.argmax(pred_crnn_np,axis=1)]
        res_str = decode(chars) # final text

        text_list.append(res_str)

    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    '''
    path = '/usr/share/fonts/truetype/nanum/NanumGothicEco.ttf'
    font_name = fm.FontProperties(fname=path, size=16).get_name()
   
    plt.rc('font', family=font_name)
    mpl.rcParams['axes.unicode_minus'] = False
    fm._rebuild()
    
    plt.figure(figsize=(15,15))
    ax = plt.gca()
    im = plt.gci()
    plt.imshow(input_seglink2)
    for t in range(len(text_list)):
        ax.add_patch(plt.Polygon(xy_raw[t], fill=False, edgecolor='y', linewidth=1))
        ax.text(xy_raw[t][0][0], xy_raw[t][0][1], text_list[t], bbox={'facecolor':'y', 'alpha':0.5}, rotation=0)

    plt.axis('off')
    plt.show()
    '''
    #print('crnn predict end')
    
    return text_list