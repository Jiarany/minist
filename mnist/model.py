import tensorflow as tf


# Y=W*x +b
def regression (x):
    W = tf.Variable(tf.zeros([784, 10]), name="W")
    b = tf.Variable(tf.zeros([10]), name="b")
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    return y, [W, b]

