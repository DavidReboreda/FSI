import gzip
import cPickle

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


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

"""
plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print train_y[57]
"""

# TODO: the neural net!!
train_y = one_hot(train_y, 10)
valid_y = one_hot(valid_y, 10)
test_y = one_hot(test_y, 10)


x = tf.placeholder("float" , [None, 784])
y_ = tf.placeholder("float", [None, 10])

W = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
b = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

y = tf.nn.softmax(tf.matmul(x, W) + b)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)  # learning rate: 0.01

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
array_error = np.zeros(6)
array_salida = []
outfile = open('mnist.txt', 'w')  # Indicamos el valor 'w'.

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
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    #sess.run(loss, feed_dict={x: x_val, y_: y_val})

    error = sess.run(loss, feed_dict = {x: valid_x, y_:valid_y})
    array_salida.append(error)
    outfile.write(str(epoch) + ' -> ' + str(error) + '\n')
    print "Epoch #:", epoch, "Error: ", error
    result = sess.run(y, feed_dict={x: batch_xs})
    count_val = 0.0
    """
    
    for b, r in zip(batch_ys, result):
        print b, "-->", r
        if np.argmax(b) == np.argmax(r):
            count_val += 1
    print "Porcentaje de acierto: ", (count_val/len(batch_ys))*100, "%"
    print "----------------------------------------------------------------------------------"
    """
    epoch = epoch + 1
    n = epoch % 6
    array_error[n] = error
    if epoch >= 6:
        continua = error_calc(array_error)
    if error < 0.01:
        break

error_test = sess.run(loss, feed_dict = {x: test_x, y_:test_y})
print "Error de conjunto de test: ", error_test
outfile.write('Error de test --> ' + str(error_test) + '\n')
result = sess.run(y, feed_dict={x: test_x})
count_test = 0.0
for b, r in zip(test_y, result):
    #print b, "-->", r.round(0)
    if np.argmax(b) == np.argmax(r):
        count_test += 1
print "Aciertos en conjunto test: ", count_test
outfile.write("Aciertos en conjunto test: " + str(count_test) + "\n")
print "Errores en conjunto test: ", len(test_y)-count_test
outfile.write("Errores en conjunto test: " + str(len(test_y) - count_test) + "\n")
print "Porcentaje de acierto: ", (count_test/len(test_y))*100, "%"
outfile.write("Porcentaje de acierto: " + str((count_test/len(test_y))*100) + "%\n")
print "----------------------------------------------------------------------------------"

outfile.close()


plt.plot(array_salida)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
