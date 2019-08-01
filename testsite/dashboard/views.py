from django.shortcuts import render

import argparse
import json
import numpy as np
import requests
from keras.applications import inception_v3
from keras.preprocessing import image


def predict(request):
    '''
    TODO : 외부 이미지를 Response를 받아서 처리

    # Argument parser 형태로 이미지를 받음.
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-i", "--image", required=True,
    #                help="path of the image")
    #args = vars(ap.parse_args())

    #image_path = args['image']
    '''

    # 테스트를 위해 임의의 로컬 이미지를 가져옴.
    image_path = "/home/primadonna/Documents/django/tfserver/test_images/car.jpg"

    # 이미지 전처리
    img = image.img_to_array(image.load_img(image_path, target_size=(224, 224))) / 255.

    # this line is added because of a bug in tf_serving(1.10.0-dev)
    img = img.astype('float16')

    payload = {
        "instances": [{'input_image': img.tolist()}]
    }

    # TensorFlow Serving server에 Post request를 보냄
    r = requests.post('http://localhost:9000/v1/models/ImageClassifier:predict', json=payload)
    pred = json.loads(r.content.decode('utf-8'))

    prediction = json.dumps(inception_v3.decode_predictions(np.array(pred['predictions']))[0])
    # Response decoding
    # decode_predictions(preds, top=5) by default gives top 5 results
    # You can pass "top=10" to get top 10 predicitons
    print(prediction)

    return render(request, "predict.html", {'prediction': prediction})
