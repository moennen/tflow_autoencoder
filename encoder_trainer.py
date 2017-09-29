'''
Trainer
'''

from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf

import encoder_dataprovider as dataprovider
import encoder_model as model

#
# Set the training parameters
#
dropout = 0.75
training_iters = 200000
log_step = 10
# data_size should be provided by the data
data_size = [128, 128, 3]
batch_size = 128

#
# Create training datasets
#
train_dataset = dataprovider.read_dataset("/train_dataset_file_lst", data_size)
sample_train_dataset = dataprovider.read_dataset("/train_dataset_file_lst", data_size)
test_dataset = dataprovider.read_dataset("/test_dataset_file_lst", data_size)

#
# Create the model
#
model, cost = model.create_model(data_size, batch_size)

# tf Graph input
ref = tf.placeholder(tf.float32, shape=data_size, name="reference_image")
probe = tf.placeholder(tf.float32, shape=data_size, name="probe_image")
flow = tf.placeholder(tf.float32, shape=data_size, name="flow_image")

dropout_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

#
# Define loss and optimizer
#
learning_rate = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Set the model accuracy as simple mean square
accuracy = tf.reduce_mean(tf.square(tf.substract(model, pred)))

#
# Initializing persistency
#
persistency = tf.train.Saver()

#
# Initializing the board
#
tf.summary.scalar("model_loss", cost)
tf.summary.scalar("model_accuracy", accuracy)
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
grads = tf.gradients(cost, tf.trainable_variables())
grads = list(zip(grads, tf.trainable_variables()))
for grad, var in grads:
    tf.summary.histogram(var.name + '/gradient', grad)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(logs_path,
                                       graph=tf.get_default_graph())

#
# Initializing the variables
#
#init = tf.global_variables_initializer()
init = tf.contrib.layers.xavier_initializer()
log_train_cost_accuracy = np.zeros(n_steps, 2)
log_test_cost_accuracy = np.zeros(n_steps, 2)

#
# Launch the graph
#
with tf.Session() as sess:
    sess.run(init)
    # Restore model if needed
    persistency.restore(sess, input_path)

    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        # Get the next training batch
        batch_img_1, batch_img_2, batch_flow = train_dataset.get_next(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={learning_rate: 0.001,
                                       img_1: batch_img_1,
                                       img_2: batch_img_2,
                                       flow: batch_flow,
                                       keep_prob: dropout})
        # Log
        if step % log_step == 0:

            # Calculate train loss and accuracy
            batch_img_1, batch_img_2, batch_flow = sample_train_dataset.get()
            loss, acc, summary = sess.run([cost, accuracy, merged_summary_op],
                                          feed_dict={img_1: batch_img_1,
                                                     img_2: batch_img_2,
                                                     flow: batch_flow,
                                                     keep_prob: 1.})
            log_train_cost_accuracy[0, step] = acc
            log_train_cost_accuracy[1, step] = loss

            print(str(step * batch_size) + " " + "{:.6f}".format(loss) + " " +
                  "{:.5f}".format(acc))

            summary_writer.add_summary(summary, step)

            # Calculate test loss and accuracy
            batch_img_1, batch_img_2, batch_flow = test_dataset.get()
            loss, acc = sess.run([cost, accuracy], feed_dict={img_1: batch_img_1,
                                                              img_2: batch_img_2,
                                                              flow: batch_flow,
                                                              keep_prob: 1.})
            log_test_cost_accuracy[0, step] = acc
            log_test_cost_accuracy[1, step] = loss

            print(str(step * batch_size) + " " + "{:.6f}".format(loss) + " " +
                  "{:.5f}".format(acc))

            # Save current model parameters
            persistency.save(sess, output_path)

            # Plot
            plt.plot(train_X, train_Y, 'ro', label='Original data')
            plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
            plt.legend()
            plt.show()

        step += 1

# Run the command line: "tensorboard --logdir=/tmp/tensorflow_logs"
# Open http://0.0.0.0:6006/
