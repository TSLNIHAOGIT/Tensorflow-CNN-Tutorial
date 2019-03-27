#coding=utf-8

import os
#图像读取库
from PIL import Image
#矩阵运算库
import numpy as np
import tensorflow as tf
import time

# 数据文件夹
data_dir = "data"
# 训练还是测试
train = False
# 模型文件路径
model_path = "model/image_model"
# 从文件夹读取图片和标签到numpy数组中
# 标签信息在文件名中，例如1_40.jpg表示该图片的标签为1
def read_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        image = Image.open(fpath)
        data = np.array(image) / 255.0
        label = int(fname.split("_")[0])
        datas.append(data)
        labels.append(label)

    datas = np.array(datas)
    labels = np.array(labels)

    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels


fpaths, datas, labels = read_data(data_dir)



# 计算有多少类图片
num_classes = len(set(labels))
graph=tf.Graph()
with graph.as_default():

    # 定义Placeholder，存放输入和标签
    datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_placeholder = tf.placeholder(tf.int32, [None])

    # 存放DropOut参数的容器，训练时为0.25，测试时为0
    dropout_placeholdr = tf.placeholder(tf.float32)

    # 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活#卷积核是对所有通道进行卷积操作的
    conv0 = tf.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

    # 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
    conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

    # 将3维特征转换为1维向量
    flatten = tf.layers.flatten(pool1)

    # 全连接层，转换为长度为100的特征向量
    fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

    # 加上DropOut，防止过拟合
    dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

    # 未激活的输出层
    logits = tf.layers.dense(dropout_fc, num_classes)

    predicted_labels = tf.arg_max(logits, 1)

    # 利用交叉熵定义损失
    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(labels_placeholder, num_classes),
        logits=logits
    )
    # 平均损失
    mean_loss = tf.reduce_mean(losses)

    # # 定义优化器，指定要优化的损失函数
    # optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)

    correct_prediction = tf.equal(tf.cast(predicted_labels, tf.int32), labels_placeholder)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # init = tf.initialize_all_variables()



#
# print("logits: ", logits)
# print("mean_loss: ", mean_loss)


#打乱顺序
num_example=datas.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
datas=datas[arr]
labels=labels[arr]

#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


with tf.Session(
        config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True),
        graph=graph) as sess:
    # 用于保存和载入模型#要放在session里面才可以
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    n_epoch = 1
    batch_size = 100

    # sess.run(init)
    for epoch in range(n_epoch):
        start_time = time.time()
        # validation
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(datas, labels, batch_size, shuffle=False):
            err, ac = sess.run([mean_loss, acc], feed_dict={datas_placeholder: x_val_a, labels_placeholder: y_val_a,dropout_placeholdr: 0})
            val_loss += err;
            val_acc += ac;
            n_batch += 1
        print("   validation loss: %f" % (val_loss / n_batch))
        print("   validation acc: %f" % (val_acc / n_batch))

    # if train:
    #     print("训练模式")
    #     # 如果是训练，初始化参数
    #     sess.run(init)
    #     # 定义输入和Label以填充容器，训练时dropout为0.25
    #     train_feed_dict = {
    #         datas_placeholder: datas,
    #         labels_placeholder: labels,
    #         dropout_placeholdr: 0.25
    #     }
    #     for step in range(150):
    #         _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)
    #
    #         if step % 10 == 0:
    #             print("step = {}\tmean loss = {}".format(step, mean_loss_val))
    #     saver.save(sess, model_path)
    #     print("训练结束，保存模型到{}".format(model_path))
    # else:
    #     print("测试模式")
    #     # 如果是测试，载入参数
    #     saver.restore(sess, model_path)
    #     print("从{}载入模型".format(model_path))
    #     # label和名称的对照关系
    #     label_name_dict = {
    #         0: "飞机",
    #         1: "汽车",
    #         2: "鸟"
    #     }
    #     # 定义输入和Label以填充容器，测试时dropout为0
    #     test_feed_dict = {
    #         datas_placeholder: datas,
    #         labels_placeholder: labels,
    #         dropout_placeholdr: 0
    #     }
    #     predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
    #     # 真实label与模型预测label
    #     for fpath, real_label, predicted_label in zip(fpaths, labels, predicted_labels_val):
    #         # 将label id转换为label名
    #         real_label_name = label_name_dict[real_label]
    #         predicted_label_name = label_name_dict[predicted_label]
    #         print("{}\t{} => {}".format(fpath, real_label_name, predicted_label_name))

#









