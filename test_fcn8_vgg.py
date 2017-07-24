#!/usr/bin/env python

import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

import fcn8_vgg
import utils
import loss

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from tensorflow.python.framework import ops

# img1 = scp.misc.imread("/home/white/下载/VOC2012/JPEGImages/2007_000123.jpg")
# y_=np.loadtxt('/home/white/下载/VOC2012/SegTXT/2007_000123.txt')

with tf.Session() as sess:
    images = tf.placeholder("float")
    labels=tf.placeholder("int32")
    # feed_dict = {images: img1,labels:y_}   #单独一张图片时
    batch_images = tf.expand_dims(images, 0)
    # global_step = tf.Variable(0, name='global_step', trainable=False)  #单独一张图片时

    vgg_fcn = fcn8_vgg.FCN8VGG()
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images, debug=True)

    print('Finished building Network.')

    logging.warning("Score weights are initialized random.")
    logging.warning("Do not expect meaningful results.")

    # logging.info("Start Initializing Variabels.")

    saver=tf.train.Saver()

    # labels=np.ones([489,368])


    print('Running the Network')
    # tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
    # down, up = sess.run(tensors, feed_dict=feed_dict)
    # logits = tf.reshape(vgg_fcn.upscore32, (-1, 20))
    logits=vgg_fcn.upscore32
    # softmax=tf.nn.softmax(logits)
    # init = tf.constant_initializer(value=labels,dtype=tf.float32)
    # labels2=tf.get_variable(name='labels',initializer=init,shape=[489,368])
    labels2=tf.expand_dims(labels, 0)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels2,
                                                                   logits=logits,
                                                                   )
    cross_entropy=tf.reduce_mean(cross_entropy)
    optimizer = tf.train.GradientDescentOptimizer(0.000001)
    train_op = optimizer.minimize(cross_entropy)
    # init = tf.global_variables_initializer()
    # sess.run(init)


    ###################################################训练所有图片
    root_path_x = '/home/white/下载/VOC2012/JPEGImages/'
    root_path_l = '/home/white/下载/VOC2012/SegTXT/'
    train_txt = '/home/white/下载/VOC2012/ImageSets/Segmentation/train.txt'
    train_file = open(train_txt)
    pic_name = train_file.readline()
    while pic_name:
        temp_picN = root_path_x + pic_name[0:-1]+'.jpg'
        temp_lab = root_path_l + pic_name[0:-1]+'.txt'
        img1 = scp.misc.imread(temp_picN)
        y_ = np.loadtxt(temp_lab)
        feed_dict = {images: img1, labels: y_}
        for step in range(100):
            sess.run(train_op, feed_dict=feed_dict)
        up, loss = sess.run([vgg_fcn.pred_up, cross_entropy], feed_dict=feed_dict)
        print('loss:%0.04f' % loss)
        up_color = utils.color_image(up[0])
        img_save_path='/home/white/下载/VOC2012/my_train/'+pic_name[0:-1]+'.png'
        scp.misc.imsave(img_save_path, up_color)
        pic_name = train_file.readline()
    save_path = saver.save(sess, '/tmp/model1.ckpt')

    ###################################################


    ###########################################################训练一张图片
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # _, logits2, loss = sess.run([train_op, logits, cross_entropy], feed_dict=feed_dict)

    # for step in range(100):
    #     # print('step:%d'%step)
    #     _,loss=sess.run([train_op,cross_entropy],feed_dict=feed_dict)
    #     if step%10==9:
    #         print('step:%d' % step)
    #         print('loss:%0.04f'%loss)
    # up,loss=sess.run([vgg_fcn.pred_up,cross_entropy], feed_dict=feed_dict)
    # save_path=saver.save(sess,'/tmp/model.ckpt')

    # down_color = utils.color_image(down[0])
    ###########################################################



    ###########################################################恢复/加载模型
    saver.restore(sess,'/tmp/model1.ckpt')
    print('model restored')
    img1 = scp.misc.imread("/home/white/下载/VOC2012/JPEGImages/2007_000032.jpg")
    y_=np.loadtxt('/home/white/下载/VOC2012/SegTXT/2007_000032.txt')
    feed_dict = {images: img1, labels: y_}
    up, loss = sess.run([vgg_fcn.pred_up, cross_entropy], feed_dict=feed_dict)
    up_color = utils.color_image(up[0])
    scp.misc.imsave('fcn8_test_from_model.png', up_color)
    print(loss)
    ###########################################################



    # up_color = utils.color_image(up[0])
    # #
    # # scp.misc.imsave('fcn8_downsampled.png', down_color)
    # scp.misc.imsave('fcn8_test_from_model.png', up_color)
print('end')

# root_path_x='/home/white/下载/VOC2012/JPEGImages/'
# root_path_l='/home/white/下载/VOC2012/SegTXT/'
# train_txt='/home/white/下载/VOC2012/ImageSets/Segmentation/train.txt'
# train_file=open(train_txt)
# pic_name=train_file.readline()
# while pic_name:
#     temp_picN=root_path_x+pic_name[0:-1]
#     temp_lab=root_path_l+pic_name[0:-1]
#     img1 = scp.misc.imread(temp_picN)
#     y_ = np.loadtxt(temp_lab)
#     feed_dict = {images: img1, labels: y_}

