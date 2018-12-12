"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.data import Iterator
import data_provider as dataset
from validation_function2 import reli

"""
Configuration Part.
"""
# Path to the textfiles for the trainings and validation set
train_file = 'data/cifar10/fulltrain.txt'
val_file = 'data/cifar10/fulltest_head1000.txt'
data_dir = 'data/cifar10'

# Learning params
learning_rate = 0.01
num_epochs = 1250
batch_size = 128

# Network params
dropout_rate = 0.75
num_classes = 10
train_layers = ['fc8', 'fclat', 'fc7', 'fc6']

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "tmp/finetune_alexnet/tensorboard"
checkpoint_path = "tmp/finetune_alexnet/checkpoints"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.makedirs(os.path.join(os.getcwd(), checkpoint_path))

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 data_dir=data_dir,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  data_dir=data_dir,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8
embeddings = tf.round(model.fclat)
embeddings_floats = model.fc7

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=score,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):

    gst = tf.train.create_global_step()

    # Get gradients of all trainable variables
    optimiser = tf.train.GradientDescentOptimizer(learning_rate)

    grads_and_vars = optimiser.compute_gradients(loss, var_list)
    #gradients = tf.gradients(loss, var_list)
    #gradients = list(zip(gradients, var_list))

    fc6w_grad, _ = grads_and_vars[-8]
    fc6b_grad, _ = grads_and_vars[-7]
    fc7w_grad, _ = grads_and_vars[-6]
    fc7b_grad, _ = grads_and_vars[-5]
    fclatw_grad, _ = grads_and_vars[-4]
    fclatb_grad, _ = grads_and_vars[-3]
    fc8w_grad, _ = grads_and_vars[-2]
    fc8b_grad, _ = grads_and_vars[-1]

    # Create optimizer and apply gradient descent to the trainable variables
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimiser.apply_gradients([(fc6w_grad, var_list[0]),
                                        (fc6b_grad, var_list[1]),
                                        (fc7w_grad, var_list[2]),
                                        (fc7b_grad, var_list[3]),
                                        (fclatw_grad*10, var_list[4]),
                                        (fclatb_grad*10, var_list[5]),
                                        (fc8w_grad, var_list[6]),
                                        (fc8b_grad, var_list[7])], global_step=gst)

# Add gradients to summary
for gradient, var in grads_and_vars:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        database_emb = []
        database_lab = []
        database_embf = []

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)
            
            # And run the training op
            _, emb, embf = sess.run([train_op, embeddings, embeddings_floats], feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate})

            database_emb.extend(emb.tolist())
            database_lab.extend(label_batch)
            database_embf.extend(embf.tolist())

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch*train_batches_per_epoch + step)

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        val_emb = []
        val_lab = []
        val_embf = []
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            acc, emb, embf = sess.run([accuracy, embeddings, embeddings_floats], feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
            val_emb.extend(emb.tolist())
            val_lab.extend(label_batch)
            val_embf.extend(embf.tolist())
        test_acc /= test_count
        #val_img, val_lab = val_data.all_data
        val_emb = np.array(val_emb)
        val_embf = np.array(val_embf)
        val_lab = np.array(val_lab)

        database_emb = np.array(database_emb)
        database_embf = np.array(database_embf)
        database_lab = np.array(database_lab)
        '''
        my_dict = {}
        my_dict['val_emb'] = val_emb
        my_dict['val_embf'] = val_embf
        my_dict['val_lab'] = val_lab
        my_dict['database_emb'] = database_emb
        my_dict['database_embf'] = database_embf
        my_dict['database_lab'] = database_lab

        try:
            np.save('model_data/trained_weights/data.npy', np.array(my_dict))
        except:
            print("couldn't save")
        '''
        print("Precision computation")
        print("{} Rel(i) Validation Accuracy(3) = {:.4f}".format(datetime.now(),
                                                       reli(3, 120, val_emb, val_embf, val_lab, database_emb, database_embf, database_lab)))
        print("{} Rel(i) Validation Accuracy(6) = {:.4f}".format(datetime.now(),
                                                       reli(6, 120, val_emb, val_embf, val_lab, database_emb, database_embf, database_lab)))
        print("{} Rel(i) Validation Accuracy(12) = {:.4f}".format(datetime.now(),
                                                       reli(12, 120, val_emb, val_embf, val_lab, database_emb, database_embf, database_lab)))
        print("{} Rel(i) Validation Accuracy(24) = {:.4f}".format(datetime.now(),
                                                       reli(24, 120, val_emb, val_embf, val_lab, database_emb, database_embf, database_lab)))

        print("{} Softmax Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        #print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        #checkpoint_name = os.path.join(checkpoint_path,
        #                               'model_epoch'+str(epoch+1)+'.ckpt')
        #save_path = saver.save(sess, checkpoint_name)
        
        model_dict = {}
        for layer in model.deep_params:
            model_dict[layer] = sess.run(model.deep_params[layer])

        print("saving model to %s" % 'model_data/trained_weights/weights_biases.npy')
        folder = os.path.dirname('model_data/trained_weights')
        if os.path.exists(folder) is False:
            os.makedirs(folder)

        np.save('model_data/trained_weights/weights_biases.npy', np.array(model_dict))

        print("Model saved at {}".format(datetime.now()))
