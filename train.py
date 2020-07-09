import os
import cv2
import random
import tensorflow as tf
import numpy as np
from pre_do import compute_iou, load_gt_boxes, wandhG, compute_regression,image_cut
from rpn_model import RPNplus
import glob
pos_thresh = 0.5
neg_thresh = 0.1
grid_width = grid_height = 16
image_height, image_width = 720, 960
def encode_label(gt_boxes):
    target_scores = np.zeros(shape=[45, 60, 9, 2])
    target_bboxes = np.zeros(shape=[45, 60, 9, 4])
    target_masks  = np.zeros(shape=[45, 60, 9])
    for i in range(45):
        for j in range(60):
            for k in range(9):
                center_x = j * grid_width + grid_width * 0.5
                center_y = i * grid_height + grid_height * 0.5
                xmin = center_x - wandhG[k][0] * 0.5
                ymin = center_y - wandhG[k][1] * 0.5
                xmax = center_x + wandhG[k][0] * 0.5
                ymax = center_y + wandhG[k][1] * 0.5
                if (xmin > -5) & (ymin > -5) & (xmax < (image_width+5)) & (ymax < (image_height+5)):
                    anchor_boxes = np.array([xmin, ymin, xmax, ymax])
                    anchor_boxes = np.expand_dims(anchor_boxes, axis=0)
                    ious = compute_iou(anchor_boxes, gt_boxes)
                    positive_masks = ious >= pos_thresh
                    negative_masks = ious <= neg_thresh
                    if np.any(positive_masks):
                        target_scores[i, j, k, 1] = 1.
                        target_masks[i, j, k] = 1
                        max_iou_idx = np.argmax(ious)
                        selected_gt_boxes = gt_boxes[max_iou_idx]
                        target_bboxes[i, j, k] = compute_regression(selected_gt_boxes, anchor_boxes[0])
                    if np.all(negative_masks):
                        target_scores[i, j, k, 0] = 1.
                        target_masks[i, j, k] = -1
    return target_scores, target_bboxes, target_masks
def process_image_label(image_path, label_path):
    raw_image1 = cv2.imread(image_path)
    raw_image=image_cut((raw_image1))
    gt_boxes = load_gt_boxes(label_path)
    target = encode_label(gt_boxes)
    return raw_image/255., target
def create_image_label_path_generator(synthetic_dataset_path):
    image_num = 95
    image_label_paths = [(os.path.join(synthetic_dataset_path, "image/%d.jpg" %(idx+1)),
                          os.path.join(synthetic_dataset_path, "imageAon/%d.txt"%(idx+1))) for idx in range(image_num)]
    while True:
        random.shuffle(image_label_paths)
        for i in range(image_num):
            yield image_label_paths[i]
def DataGenerator(synthetic_dataset_path, batch_size):
    image_label_path_generator = create_image_label_path_generator(synthetic_dataset_path)
    while True:
        images = np.zeros(shape=[batch_size, image_height, image_width, 3], dtype=np.float)
        target_scores = np.zeros(shape=[batch_size, 45, 60, 9, 2], dtype=np.float)
        target_bboxes = np.zeros(shape=[batch_size, 45, 60, 9, 4], dtype=np.float)
        target_masks  = np.zeros(shape=[batch_size, 45, 60, 9], dtype=np.int)
        for i in range(batch_size):
            image_path, label_path = next(image_label_path_generator)
            image, target = process_image_label(image_path, label_path)
            images[i] = image
            target_scores[i] = target[0]
            target_bboxes[i] = target[1]
            target_masks[i]  = target[2]
        yield images, target_scores, target_bboxes, target_masks
def compute_loss(target_scores, target_bboxes, target_masks, pred_scores, pred_bboxes):
    score_loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_scores, logits=pred_scores)
    foreground_background_mask = (np.abs(target_masks) == 1).astype(np.int)
    score_loss = tf.reduce_sum(score_loss * foreground_background_mask, axis=[1,2,3]) / np.sum(foreground_background_mask)
    score_loss = tf.reduce_mean(score_loss)
    boxes_loss = tf.abs(target_bboxes - pred_bboxes)
    boxes_loss = 0.5 * tf.pow(boxes_loss, 2) * tf.cast(boxes_loss<1, tf.float32) + (boxes_loss - 0.5) * tf.cast(boxes_loss >=1, tf.float32)
    boxes_loss = tf.reduce_sum(boxes_loss, axis=-1)
    foreground_mask = (target_masks > 0).astype(np.float32)
    boxes_loss = tf.reduce_sum(boxes_loss * foreground_mask, axis=[1,2,3]) / np.sum(foreground_mask)
    boxes_loss = tf.reduce_mean(boxes_loss)
    return score_loss, boxes_loss
EPOCHS = 10
STEPS = 4000
batch_size = 2
lambda_scale = 1.
synthetic_dataset_path="/home/xuyangfan/下载/data"
TrainSet = DataGenerator(synthetic_dataset_path, batch_size)
model = RPNplus()
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
writer = tf.summary.create_file_writer("./log")
global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)
for epoch in range(EPOCHS):
    for step in range(STEPS):
        global_steps.assign_add(1)
        image_data, target_scores, target_bboxes, target_masks = next(TrainSet)
        with tf.GradientTape() as tape:
            pred_scores, pred_bboxes = model(image_data)
            score_loss, boxes_loss = compute_loss(target_scores, target_bboxes, target_masks, pred_scores, pred_bboxes)
            total_loss = score_loss + lambda_scale * boxes_loss
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print("=> epoch %d  step %d  total_loss: %.6f  score_loss: %.6f  boxes_loss: %.6f" %(epoch+1, step+1,
                                                        total_loss.numpy(), score_loss.numpy(), boxes_loss.numpy()))
        # writing summary data
        with writer.as_default():
            tf.summary.scalar("total_loss", total_loss, step=global_steps)
            tf.summary.scalar("score_loss", score_loss, step=global_steps)
            tf.summary.scalar("boxes_loss", boxes_loss, step=global_steps)
        writer.flush()
    model.save_weights("RPN.h5")
#cnn对猫狗进行分类部分
tf.device('/gpu:0')
image_path=glob.glob('/home/xuyangfan/下载/data/train/*')
train_image_lable=[int(p.split('/')[3].split('.')[0]=='cat') for p in image_path]
def _pre_read(path,lable):
    image=tf.io.read_file(path)
    image=tf.image.decode_jpeg(image,channels=3)
    image=tf.image.resize(image,[256,256])
    image=tf.cast(image,tf.float32)
    image=image/255
    lable=tf.reshape(lable,[1])
    return image,lable
train_image_dataset=tf.data.Dataset.from_tensor_slices((image_path,train_image_lable))
AUTOTUNE=tf.data.experimental.AUTOTUNE
train_image_dataset=train_image_dataset.map(_pre_read,num_parallel_calls=AUTOTUNE)
batch=1
train_count=len(image_path)
train_image_dataset=train_image_dataset.shuffle(train_count).batch(batch)
train_image_dataset=train_image_dataset.prefetch(AUTOTUNE)#可以预处理一些数据以加快运行速度
model=tf.keras.Sequential([tf.keras.layers.Conv2D(64,(3,3),activation='relu'),tf.keras.layers.MaxPool2D(),
                         tf.keras.layers.Conv2D(128,(3,3),activation='relu'),tf.keras.layers.MaxPool2D(),
                         tf.keras.layers.Conv2D(256,(3,3),activation='relu'),tf.keras.layers.MaxPool2D(),
                         tf.keras.layers.Conv2D(512,(3,3),activation='relu'),tf.keras.layers.MaxPool2D(),
                         tf.keras.layers.Conv2D(1024,(3,3),activation='relu'),tf.keras.layers.GlobalAveragePooling2D(),
                         tf.keras.layers.Dense(256,activation='relu'),
                         tf.keras.layers.Dense(1)])#可以不激活通过判断结果的正负来取结果
ls=tf.keras.losses.BinaryCrossentropy()#计算二元交叉熵
optimizer=tf.keras.optimizers.Adam()
epoch_loss_avg=tf.keras.metrics.Mean('train_loss')
train_accuravy=tf.keras.metrics.Accuracy()
def train_step(model,images,lables):
    with tf.GradientTape() as t:
        pred=model(images)
        loss_step=tf.keras.losses.BinaryCrossentropy(from_logits=True)(lables,pred)
    grads=t.gradient(loss_step,model.trainable_variables)#后者的意思是可训练的参数
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    epoch_loss_avg(loss_step)
    train_accuravy(lables,tf.cast(pred>0,tf.int32))
epoch_num=10
train_loss_results=[]
train_acc_results=[]
for epoch in range(epoch_num):
    for imgs_,lables_ in train_image_dataset:
        train_step(model,imgs_,lables_)
        print('.',end='')
    print()
    train_loss_results.append(epoch_loss_avg.result())
    train_acc_results.append(train_accuravy)
    print('Epoch{} loss is {},accuracy is {}'.format(epoch+1, epoch_loss_avg.result(), train_accuravy.result()))
    epoch_loss_avg.reset_states()
    train_accuravy.reset_states()
model.save("cnn.h5")