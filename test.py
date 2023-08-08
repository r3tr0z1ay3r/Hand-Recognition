from keras.preprocessing import image
from keras.models import load_model
from keras.utils import load_img,img_to_array
import numpy as np
from numpy import argmax
model = load_model('model.h5')
print("Model Loaded Successfully")

def classify(img_file):
    img_name = img_file
    test_image = load_img(img_name, target_size = (256, 256),grayscale=True)

    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    arr = np.array(result[0])
    print(arr)
    maxx = np.amax(arr)
    max_prob = np.argmax(arr,axis=0)
    #max_prob = max_prob + 1
    classes=["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
    result = classes[max_prob-1]
    print(img_name,result)


import os
path = 'F:\\Python projects(Pycharm)\\Pantech Course stuff\\Hand Recognition\\HandGestureDataset\\test\\TWO'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
   for file in f:
     if '.png' in file:
       files.append(os.path.join(r, file))

for f in files:
   classify(f)
   print('\n')