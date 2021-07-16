import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=str)
args = parser.parse_args()

folder_path = args.folder # image folder

# folder_path = "final/test_dir/prefix/"
model_path = 'models/model/' # model folder
img_width, img_height = 384, 128

model = load_model(model_path) # load the trained model
print("Model Loaded. Loading images and predicting now ...")

images = []
files = os.listdir(folder_path)
cnt = 0
final = [["Image_Name", "Label"]]
class_names = ['infix', 'postfix', 'prefix']
2
for im in files:
    img = os.path.join(folder_path, im)
    img = image.load_img(img, target_size=(img_height, img_width),color_mode='grayscale')
    img = image.img_to_array(img)
    img /= 255.
    img = np.expand_dims(img, axis=0)
    images.append(img)
    images = np.vstack(images)
    classes = list(np.argmax(model.predict(images), axis=-1))
    final.append([im, class_names[classes[0]]])
    images = []
    cnt+=1
    if (cnt % 1000 == 0): print(cnt, "images loaded")


# stack up images list to pass for prediction
'''
images = np.vstack(images)
classes = np.argmax(model.predict(images), axis=-1)
print("Completed Prediction")
k = list(classes)
for i in range(len(files)):
    final.append([files[i], class_names[k[i]]])
'''

with open("cosmos_1.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(final)


print("====COMPLETE====")


