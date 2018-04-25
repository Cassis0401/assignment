#amazon=source and dslr=target
#https://github.com/JJMUSA/Domain-Adaptation-With-SNES/blob/master/officeData.py
import os
from PIL import Image
import numpy as np
from utils import *
import cv2
amazonPATH='./Original_images/amazon/images'
dslrPATH='./Original_images/dslr/images'
def get_image_list(dic):
    labels_list = {'printer': 21,
                   'projector': 22,
                   'file_cabinet': 9,
                   'ruler': 25,
                   'trash_can': 30,
                   'phone': 20,
                   'bottle': 4,
                   'laptop_computer': 12,
                   'bookcase': 3,
                   'letter_tray': 13,
                   'back_pack': 0,
                   'paper_notebook': 18,
                   'calculator': 5,
                   'desk_chair': 7,
                   'mug': 17,
                   'pen': 19,
                   'monitor': 15,
                   'mouse': 16,
                   'desktop_computer': 6,
                   'ring_binder': 24,
                   'stapler': 28,
                   'bike_helmet': 2,
                   'tape_dispenser': 29,
                   'desk_lamp': 8,
                   'keyboard': 11,
                   'bike': 1,
                   'punchers': 23,
                   'mobile_phone': 14,
                   'scissors': 26,
                   'headphones': 10,
                   'speaker': 27}
    image_list = []
    image_label = []
    for root, dirs, files in os.walk(dic):
        for file in files:
            if file.endswith('.jpg'):
                # print file
                temp = str(root).split('\\')
                for i in labels_list:
                    if temp[-1]==i:
                        temp2=labels_list[i]
                        label = temp2
                im = Image.open(root + '/' + file)
                im=np.array(im)
				
                image_list.append(im)
                image_label.append(label)
                #print(image_label)

    return np.array(image_list), np.array(image_label)

if __name__ == '__main__':
    image_data = cv2.imread('./Original_images/amazon/images/ruler/frame_0001.jpg')
    print(image_data)
amazonImage,amazonLabels=get_image_list(amazonPATH)
dslrImages,dslrLabels=get_image_list(dslrPATH)

x_train=amazonImage
y_train=amazonLabels
x_test=dslrImages
y_test=dslrLabels

x_domain=np.concatenate((x_train,x_test), axis =0)
y_domain=np.concatenate((y_train,y_test), axis =0)
y_domain = np.concatenate((np.zeros(y_train.shape[0]), np.ones(mnistm_train2.shape[0])),axis=0)
print(x_domain.shape)
print(y_domain.shape)