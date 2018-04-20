import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle

class dataWrapper:
    def __init__(self,data):
        self.size=len(data)
        self.x_title=[]
        self.x_body=[]
        self.y=[]
        self.seqlen_title=[]
        self.seqlen_body=[]
        self.current_batch=0
        for sample in data:
            self.x_title.append(sample[0])
            self.x_body.append(sample[1])
            self.y.append(sample[2])
            self.seqlen_title.append(len(sample[0]))
            self.seqlen_body.append(len(sample[1]))
        max_seqlen=max(self.seqlen_body+self.seqlen_title)
        #padding the samples with zero vectors
        for i in range(len(self.x_title)):
            self.x_title[i]+=[[0]*50]*(max_seqlen-len(self.x_title[i]))
            self.x_body[i]+=[[0]*50]*(max_seqlen-len(self.x_body[i]))
    # return the next batch of the data from the data set.
    def next(self,batch_size):
        if self.current_batch+batch_size<self.size:
            self.current_batch+=batch_size
            return self.x_title[self.current_batch:self.current_batch+batch_size],self.x_body[self.current_batch:self.current_batch+batch_size],self.y[self.current_batch:self.current_batch+batch_size],self.seqlen_title[self.current_batch:self.current_batch+batch_size],self.seqlen_body[self.current_batch:self.current_batch+batch_size]
        else:
            temp=self.current_batch
            self.current_batch=self.current_batch+batch_size-self.size
            batch_x_title=self.x_title[temp:]+self.x_title[:self.current_batch]
            batch_x_body=self.x_body[temp:]+self.x_body[:self.current_batch]
            batch_y=self.y[temp:]+self.y[:self.current_batch]
            batch_seqlen_title=self.seqlen_title[temp:]+self.seqlen_title[:self.current_batch]
            batch_seqlen_body=self.seqlen_body[temp:]+self.seqlen_body[:self.current_batch]
            return batch_x_title,batch_x_body,batch_y,batch_seqlen_title,batch_seqlen_body
    # return the length of the longest sequence
    def max_seqlen(self):
        return max(self.seqlen_body+self.seqlen_title)

# loading the data
data=pickle.load(open("data_stances.p","rb"))
size=len(data)
trainset=dataWrapper(data[size//3:])
testset=dataWrapper(data[:size//3])

#network parameters
learning_rate=0.001
training_iters=100000
batch_size=128
display_step=10

seq_max_len=max(trainset.max_seqlen(),testset.max_seqlen())
n_input=50
n_hidden=60
n_classes=3
n_layers=2

x_title=tf.placeholder("float",[None,seq_max_len,n_input])
x_body=tf.placeholder("float",[None,seq_max_len,n_input])
y=tf.placeholder("float",[None,n_classes])
seqlen_title=tf.placeholder(tf.int32,[None])
seqlen_body=tf.placeholder(tf.int32,[None])


weights={'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))}
biases={'out':tf.Variable(tf.random_normal([1,n_classes]))}

def dynamicRNN(x_title,x_body,seqlen_title,seqlen_body,weights,biases):

    #
    print("testing_1")
    #gru_cell=tf.contrib.rnn.MultiRNNCell([gru_cell] * 2, state_is_tuple=True)
    stacked_rnn = []
    for iiLyr in range(n_layers):
        gru_cell=tf.nn.rnn_cell.GRUCell(num_units=n_hidden)
        stacked_rnn.append(tf.contrib.rnn.DropoutWrapper(gru_cell, input_keep_prob = 0.5))
    MultiLyr_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn)
    outputs_title,states_title=tf.nn.dynamic_rnn(cell=MultiLyr_cell,inputs=x_title,sequence_length=seqlen_title,dtype=tf.float32)
    print(outputs_title)


    
    
    with tf.variable_scope('scope1',reuse=None):    
        print("testing_2")
        outputs_body,states_body=tf.nn.dynamic_rnn(cell=MultiLyr_cell,inputs=x_body,sequence_length=seqlen_body,dtype=tf.float32)
        print("testing_3")
        temp1=tf.stack([tf.range(tf.shape(seqlen_title)[0]),seqlen_title-1],axis=1)
        temp2=tf.stack([tf.range(tf.shape(seqlen_body)[0]),seqlen_body-1],axis=1)
        #for output_title in outputs_title:
        #    print(outputs_title)
        #print('check')
        
        return tf.matmul(tf.multiply(tf.gather_nd(outputs_title,temp1),tf.gather_nd(outputs_body,temp2)),weights['out'])+biases['out']

pred=dynamicRNN(x_title,x_body,seqlen_title,seqlen_body,weights,biases)
cost =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()

saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    step = 1

    saver = tf.train.import_meta_graph('./stances_ckpt/gru_stances_ckpt.meta')
    saver.restore(sess, './stances_ckpt/gru_stances_ckpt')

    weights = sess.run(weights)
    biases = sess.run(biases)

    
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
