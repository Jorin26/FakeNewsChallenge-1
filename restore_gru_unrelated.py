import tensorflow as tf

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./gru_related_ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    print(sess.run(tf.global_variables()))
