# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:01:13 2019

@author: Sooram Kang

Reference: https://github.com/Ankush96/grad-cam.tensorflow/blob/master/main.py

"""

import numpy as np
import tensorflow as tf

import PIL 
from PIL import Image
import matplotlib.pyplot as plt

import vgg16

config = {
    "img_W": 224,
    "img_H": 224
}

# %%        
def get_classmap(target_class, last_layer, conv, cmap_width, cmap_height):
    # get the score for the target class before the softmax
    y_c = tf.gather(tf.transpose(last_layer), target_class)

    # compute the gradient of the score for class, 
    # with respect to feature maps of a convolutional layer
    grads = tf.gradients(y_c, conv)[0]
        
    # get the weight of each feature map
    # (average value of all the gradient values in each feature map)
    weights = tf.reduce_mean(grads, [1, 2])			#(?, n_filters)
    n_filters = grads.get_shape().as_list()[3]  # # of filters at 'conv'
    weights = tf.reshape(weights, [-1, n_filters, 1]) #(?, n_filters, 1)
    
    # resize the last conv layer
    h, w = cmap_height, cmap_width
    feature_maps = tf.image.resize_bilinear(conv, [h, w])    #(?, h, w, n_filters)
    feature_maps = tf.reshape(feature_maps, [-1, h * w, n_filters])   #(?, h*w, n_filters)
    
    # take a weighted sum of the feature maps
    classmap = tf.matmul(feature_maps, weights) #(?, h*w, 1)
    classmap = tf.reshape(classmap, [-1, h, w]) #(?, h, w) 
    
    return classmap
 
# %% util functions
def load_img(filename):
    img = Image.open(filename)

    # preprocess the img
    img0 = img.resize((config["img_W"], config["img_H"]), resample=PIL.Image.NEAREST)
    img0 = np.array(img0)
    img0 = img0 / 255.0
    img_reshaped = np.reshape(img0, [-1, config["img_W"], config["img_H"], 3])
    
    return img, img_reshaped
    
def open_sess(gpu_num):
    c = tf.ConfigProto()
    c.gpu_options.visible_device_list = gpu_num    #"0" 
    
    sess = tf.Session(config=c)
    sess.run(tf.global_variables_initializer())
    
    return sess

# %% main
def main():
    # load an img file, preprocess and reshape it
    img_original, img_reshaped = load_img('./Black_Footed_Albatross_0001_796111.jpg')

    # open a session
    sess = open_sess("0")

    # build a model
    images = tf.placeholder("float", [1, config["img_W"], config["img_H"], 3])
    
    vgg = vgg16.Vgg16()
    vgg.build(images)
    
    feed_dict = {images: img_reshaped}
    prob = sess.run(vgg.prob, feed_dict=feed_dict)
    #print(prob)

    # get the class map value
    target_class = [np.argmax(prob)]    # the class that we're interested in        
    output = vgg.fc8                    # the output(score) layer before the softmax 
    last_conv = vgg.conv5_3             # the last feature maps
    
    label = tf.placeholder(tf.int64, [None], name='label')
    classmap = get_classmap(label, output, last_conv, config["img_W"], config["img_H"])
    
    gradcam = sess.run(classmap, feed_dict={images: img_reshaped,
                                            label: target_class})
    
    # draw the class map
    plt.imshow(img_original.resize([config["img_W"], config["img_H"]]), cmap = 'gray' )
    #plt.imshow(gradcam[0], cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')    # normalized
    plt.imshow(gradcam[0], cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest', vmin=-0.4, vmax=0.4)
    plt.savefig('test.png', dpi=256)
    plt.close() 


if __name__ == '__main__':
    main()
