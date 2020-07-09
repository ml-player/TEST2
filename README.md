
# --网络介绍
1、因为我一直写不出来直接通过faster-rcnn直接输出多种判断结果的代码，然后为了赶时间我直接通过使用faster-rcnn直接把猫狗一起输出出来（不直接做分类），然后在对输出结果图片按照框选位置进行裁剪，获得输出图像，然后再把图像放入我的卷积神经网路对其分类。
2、rpn网络搭建方面，一开始我使用直接函数式API来搭建的rpn网络模型，三层卷积接一层池化，然后重复四次（参考的faster-rcnn的结构图论文）然后再通过不同的卷积得到18和36维的结果，最后返回偏移量和损失值
3、CNN网络搭建方面，直接使用最为简单的卷积网络，直接搭建的。
4、rpn网络修改方面，参照了知乎上一位大神的写法，直接通过python的类来实现rpn网络的搭建以及优化并重新命名。（本来CNN也计划这么做的，但是不知道为什么修改后会运行不了）
5、算法方面，自己实现了iou的计算、图片的大小的修改、损失值的计算（为了效果更好后面也修改采用他人算法）、图片优化以及数据的转化等。
6、图片方面，由于如nms算法等借鉴了另一位大神的代码，所以也和他一样统一了格式使用720*960
7、由于几个关键算法采用了他人的代码导致一些我本来的代码会运行错误，所以我也是对我本来已经写好的代码做了一定的同化。
8、运行方面，代码分为了 pre_do：各类算法储存 train：对我的模型进行训练 test：导入训练所得模型和参数开始预测 rpn_model：放置了我的优化过后的模型文件
9、我的电脑成功运行了rpn网络训练部分，但是在CNN网络中出现了内存不够的警告
