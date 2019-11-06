#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy import misc
import shutil
import tensorflow as tf
import warnings
from distutils.version import LooseVersion
from matplotlib import pyplot
import numpy as np
from glob import glob
import os
import helper
data_dir = './input'


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
show_n_images = 9

mnist_images = helper.get_batch(glob(os.path.join(
    data_dir, 'img_align_celeba/*.jpg'))[:show_n_images], 28, 28, 'RGB')
pyplot.figure(figsize=(5, 5))
pyplot.imshow(helper.images_square_grid(mnist_images, 'RGB'))


# ### Check the Version of TensorFlow and Access to GPU
# This will check to make sure you have the correct version of TensorFlow and access to a GPU

# In[3]:


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# ### Input
# Implement the model_inputs function to create TF Placeholders for the Neural Network. It should create the following placeholders:

# In[4]:


def model_inputs(image_width, image_height, image_channels, z_dim):
    real_input_images = tf.placeholder(
        tf.float32, [None, image_width, image_height, image_channels], 'real_input_images')
    input_z = tf.placeholder(tf.float32, [None, z_dim], 'input_z')
    learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
    return real_input_images, input_z, learning_rate


# ### Discriminator
# Implement discriminator to create a discriminator neural network that discriminates on images.

# In[5]:


def discriminator(images, reuse=False, alpha=0.2, keep_prob=0.5):

    with tf.variable_scope('discriminator', reuse=reuse):
        # Input layer is 28x28xn
        # Convolutional layer, 14x14x64
        conv1 = tf.layers.conv2d(images, 64, 5, 2, padding='same',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        lrelu1 = tf.maximum(alpha * conv1, conv1)
        drop1 = tf.layers.dropout(lrelu1, keep_prob)

        # Strided convolutional layer, 7x7x128
        conv2 = tf.layers.conv2d(drop1, 128, 5, 2, 'same', use_bias=False)
        bn2 = tf.layers.batch_normalization(conv2)
        lrelu2 = tf.maximum(alpha * bn2, bn2)
        drop2 = tf.layers.dropout(lrelu2, keep_prob)

        # Strided convolutional layer, 4x4x256
        conv3 = tf.layers.conv2d(drop2, 256, 5, 2, 'same', use_bias=False)
        bn3 = tf.layers.batch_normalization(conv3)
        lrelu3 = tf.maximum(alpha * bn3, bn3)
        drop3 = tf.layers.dropout(lrelu3, keep_prob)

        # fully connected
        flat = tf.reshape(drop3, (-1, 4*4*256))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)

        return out, logits


# ### Generator
# Implement generator to generate an image using z.

# In[6]:


def generator(z, out_channel_dim, is_train=True, alpha=0.2, keep_prob=0.5):

    with tf.variable_scope('generator', reuse=(not is_train)):
        # First fully connected layer, 4x4x1024
        fc = tf.layers.dense(z, 4*4*1024, use_bias=False)
        fc = tf.reshape(fc, (-1, 4, 4, 1024))
        bn0 = tf.layers.batch_normalization(fc, training=is_train)
        lrelu0 = tf.maximum(alpha * bn0, bn0)
        drop0 = tf.layers.dropout(lrelu0, keep_prob, training=is_train)

        # Deconvolution, 7x7x512
        conv1 = tf.layers.conv2d_transpose(
            drop0, 512, 4, 1, 'valid', use_bias=False)
        bn1 = tf.layers.batch_normalization(conv1, training=is_train)
        lrelu1 = tf.maximum(alpha * bn1, bn1)
        drop1 = tf.layers.dropout(lrelu1, keep_prob, training=is_train)

        # Deconvolution, 14x14x256
        conv2 = tf.layers.conv2d_transpose(
            drop1, 256, 5, 2, 'same', use_bias=False)
        bn2 = tf.layers.batch_normalization(conv2, training=is_train)
        lrelu2 = tf.maximum(alpha * bn2, bn2)
        drop2 = tf.layers.dropout(lrelu2, keep_prob, training=is_train)

        # Output layer, 28x28xn
        logits = tf.layers.conv2d_transpose(
            drop2, out_channel_dim, 5, 2, 'same')

        out = tf.tanh(logits)

        return out


# ### Loss
# Implement model_loss to build the GANs for training and calculate the loss.

# In[7]:


def model_loss(input_real, input_z, out_channel_dim, alpha=0.2, smooth_factor=0.1):

    d_model_real, d_logits_real = discriminator(input_real, alpha=alpha)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                labels=tf.ones_like(d_model_real) * (1 - smooth_factor)))

    input_fake = generator(input_z, out_channel_dim, alpha=alpha)
    d_model_fake, d_logits_fake = discriminator(
        input_fake, reuse=True, alpha=alpha)

    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))

    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

    return d_loss_real + d_loss_fake, g_loss


# ### Optimization
# Implement model_opt to create the optimization operations for the GANs.

# In[8]:


def model_opt(d_loss, g_loss, learning_rate, beta1):

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(
            learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(
            learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt


# ## Neural Network Training
# ### Show Output
# Use this function to show the current output of the generator during training.

# In[9]:


def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):

    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})
#    print(samples.shape)
    images_grid = helper.images_square_grid(samples, image_mode) #if want to show process of tranning uncommnet this line
    pyplot.imshow(images_grid, cmap=cmap)   #if want to show process of tranning uncommnet this line
    pyplot.axis('off')  #if want to show process of tranning uncommnet this line
    pyplot.show()  #if want to show process of tranning uncommnet this line
    return images_grid #if want to show process of tranning uncommnet this line
    #return samples  # return for generate 500 images


# ### output_fig
# After trained your GAN, generated 500 images (each image contains 3x3 grid of images) and save it by the "output_fig" function below

# In[10]:


def output_fig(images_array, file_name):
    # the shape of your images_array should be (9, width, height, 3),  28 <= width, height <= 112
    pyplot.figure(figsize=(6, 6), dpi=100)
    pyplot.imshow(helper.images_square_grid(images_array, 'RGB'))
    pyplot.axis("off")
    pyplot.savefig(file_name+'.png', bbox_inches='tight', pad_inches=0)


# ### Train
# Implement train to build and train the GANs.

# ### CelebA
# Run GANs on CelebA.

# In[11]:


# #print_every and show_every can temporary show the tranning process of GAN, also test if the model can work correctly

# In[15]:


def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode,
          print_every=2, show_every=10):

    input_real, input_z, _ = model_inputs(
        data_shape[1], data_shape[2], data_shape[3], z_dim)
    d_loss, g_loss = model_loss(input_real, input_z, data_shape[3], alpha=0.2)
    d_train_opt, g_train_opt = model_opt(d_loss, g_loss, learning_rate, beta1)

    saver = tf.train.Saver()
    sample_z = np.random.uniform(-1, 1, size=(72, z_dim))

    samples, losses = [], []

    steps = 0
    count = 0

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        module_file = tf.train.latest_checkpoint('./model/')
        #saver.restore(sess, module_file)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # sess.run(tf.global_variables_initializer())
        if os.path.exists('./output/'):
            shutil.rmtree('./output/', ignore_errors=True)

        os.mkdir('output')

        for epoch_i in range(epoch_count):
            os.mkdir('output/' + str(epoch_i))
            for batch_images in get_batches(batch_size):

                steps += 1
                batch_images *= 2.0

                # Sample random noise for G
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

                # Run optimizers
                sess.run(d_train_opt, feed_dict={
                         input_real: batch_images, input_z: batch_z})
                sess.run(g_train_opt, feed_dict={input_z: batch_z})

                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = d_loss.eval(
                        {input_real: batch_images, input_z: batch_z})
                    train_loss_g = g_loss.eval({input_z: batch_z})
                    print("Epoch {}/{} Step {}...".format(epoch_i+1, epoch_count, steps),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))

                if steps % show_every == 0:
                    count = count + 1
                    iterr = count*show_every
                    # Show example output for the generator
                    images_grid = show_generator_output(
                        sess, 9, input_z, data_shape[3], data_image_mode)
                    dst = os.path.join(".\\output\\", str(
                        epoch_i), str(iterr)+".png")
                    print(dst)
                    misc.imsave(dst, images_grid)

        saver.save(sess, './model/' + 'lastmodel')
        for i in range(500):
            generate_images = show_generator_output(
                sess, 9, input_z, data_shape[3], data_image_mode)
            print(generate_images.shape)  # should be (9, width, height, 3)
            output_fig(generate_images,
                       file_name="images/{}_image".format(str.zfill(str(i), 3)))


# In[16]:


batch_size = 64
z_dim = 100
learning_rate = 0.00025
beta1 = 0.45

epochs = 20

celeba_dataset = helper.Dataset('celeba', glob(
    os.path.join(data_dir, 'img_align_celeba/*.jpg')))
print(celeba_dataset.shape)
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
          celeba_dataset.shape, celeba_dataset.image_mode)


# In[ ]:
