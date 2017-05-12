import tensorflow as tf
import numpy as np

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
entrenamiento = int (150 * 0.7)
validacion = int(150 * 0.15) + entrenamiento

x_data = data[:entrenamiento, 0:4].astype('f4')  # the samples are the four first rows of data
x_val = data[entrenamiento:validacion, 0:4].astype('f4')
x_test = data[validacion: , 0:4].astype('f4')
y_data = one_hot(data[:entrenamiento, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code
y_val = one_hot(data[entrenamiento:validacion, 4].astype(int), 3)
y_test = one_hot(data[validacion: , 4].astype(int), 3)



print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20

#for epoch in xrange(100):
error = 100
epoch = 0
array_error = [6,5,4,3,2,1]

def error_calc (x):
    calc = False
    for i in range(len(x) - 1):
        if ((x[i] - x[i + 1]) < 0.0001):
            calc = False
        else:
            return True
    return calc

continua = True

while continua == True:
    for jj in xrange(len(x_data) / batch_size):
        batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    #sess.run(loss, feed_dict={x: x_val, y_: y_val})

    error = sess.run(loss, feed_dict = {x: x_val, y_:y_val})
    print "Epoch #:", epoch, "Error: ", error
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print b, "-->", r
    print "----------------------------------------------------------------------------------"
    epoch = epoch + 1
    n = epoch % 6
    array_error[n] = error
    if epoch >= 6:
        continua = error_calc(array_error)
    if error < 0.01:
        break

print "Error de conjunto de test: ", sess.run(loss, feed_dict = {x: x_test, y_:y_test})
result = sess.run(y, feed_dict={x: x_test})
for b, r in zip(y_test, result):
    print b, "-->", r
print "----------------------------------------------------------------------------------"