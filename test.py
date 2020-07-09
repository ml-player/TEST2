import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from rpn_model import RPNplus
from pre_do import decode_output, plot_boxes_on_image, nms
model = RPNplus()
fake_data = np.ones(shape=[1, 720, 960, 3]).astype(np.float32)
model(fake_data) # initialize model to load weights
model.load_weights("./RPN.h5")
def cnn_(img):
    cnn_model=tf.keras.models.load_model("cnn.h5")
    lable=cnn_model.predict(img)
    cor=(0,255,0)
    if(lable==1):
        cor=(0,0,255)
    return cor
image_path=("")
img_test=cv2.imread(image_path)
image_data = np.expand_dims(img_test / 255., 0)
pred_scores, pred_bboxes = model(image_data)
pred_scores = tf.nn.softmax(pred_scores, axis=-1)
pred_scores, pred_bboxes = decode_output(pred_bboxes, pred_scores, 0.9)
pred_bboxes = nms(pred_bboxes, pred_scores, 0.5)
for img_box in pred_bboxes:
    img_x=img_test[img_box[1]:img_box[1]+img_box[3],img_box[0]:img_box[0]+img_box[2]]
    color_=cnn_(img_x)
    cv2.rectangle(img_test,(img_box[0],img_box[1]),(img_box[2]+img_box[0],img_box[1]+img_box[3]),color_)
cv2.imshow(img_test)
cv2.waitkey(0)
