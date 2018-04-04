import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
from data_wrapper import DataWrapper 

# loading the data
data=pickle.load(open("data_related.p","rb"))
size=len(data)
trainset=DataWrapper(data[size//3:])
testset=DataWrapper(data[:size//3])

#network parameters
learning_rate=0.001
training_iters=100000
batch_size=128
display_step=10

seq_max_len = max(trainset.max_seqlen(), testset.max_seqlen())
n_input = 50
n_hidden = 60
n_classes = 2

x_title = tf.placeholder("float", [None, seq_max_len,n_input])
x_body = tf.placeholder("float", [None, seq_max_len,n_input])
y = tf.placeholder("float", [None, n_classes])
seqlen_title = tf.placeholder(tf.int32, [None])
seqlen_body = tf.placeholder(tf.int32, [None])


weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
biases = {'out': tf.Variable(tf.random_normal([1, n_classes]))}

def dynamicRNN(x_title, x_body, seqlen_title, seqlen_body, weights, biases):

    gru_cell = rnn.GRUCell(n_hidden)
    print("testing_1")
    outputs_title, states_title = tf.nn.dynamic_rnn(cell = gru_cell,
            inputs = x_title,
            sequence_length = seqlen_title,
            dtype = tf.float32)

    with tf.variable_scope('scope1', reuse = None):    
        print("testing_2")
        outputs_body, states_body = tf.nn.dynamic_rnn(
                cell = gru_cell,
                inputs = x_body,
                sequence_length = seqlen_body,
                dtype = tf.float32)
        print("testing_3")
        temp1 = tf.stack([tf.range(tf.shape(seqlen_title)[0]), seqlen_title - 1], axis = 1)
        temp2 = tf.stack([tf.range(tf.shape(seqlen_body)[0]), seqlen_body - 1], axis = 1)
        return tf.matmul(
                tf.multiply(
                    tf.gather_nd(outputs_title,temp1),
                    tf.gather_nd(outputs_body,temp2)),
                weights['out']) + biases['out']

pred = dynamicRNN(x_title, x_body, seqlen_title, seqlen_body, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step * batch_size < training_iters:
        print("step:", step)

        batch_x_title, batch_x_body, batch_y, batch_seqlen_title, batch_seqlen_body = trainset.next(batch_size)
        sess.run(optimizer,
                feed_dict = {
                    x_title: batch_x_title,
                    x_body: batch_x_body,
                    y: batch_y,
                    seqlen_title: batch_seqlen_title,
                    seqlen_body: batch_seqlen_body
                    })
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict = {
                x_title: batch_x_title,
                x_body: batch_x_body,
                y: batch_y,
                seqlen_title: batch_seqlen_title,
                seqlen_body: batch_seqlen_body
                })
            loss = sess.run(cost, feed_dict = {
                x_title: batch_x_title, 
                x_body: batch_x_body,
                y: batch_y,
                seqlen_title: batch_seqlen_title,
                seqlen_body: batch_seqlen_body
                })
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    save_path = saver.save(sess, "./gru_related_ckpt")
    print("Optimization Finished!")

    test_x_title = testset.x_title
    test_x_body = testset.x_body
    test_y = testset.y
    test_seqlen_title = testset.seqlen_title
    test_seqlen_body = testset.seqlen_body

    print("Test Accuracy:", sess.run(accuracy, feed_dict = {
        x_title: test_x_title,
        x_body: test_x_body,
        y: test_y,
        seqlen_title: test_seqlen_title,
        seqlen_body: test_seqlen_body
        }))
