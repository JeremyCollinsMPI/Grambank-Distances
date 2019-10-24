import tensorflow as tf
# from MakeArrays import *
import numpy as np

input_array = np.load('input_array.npy')
output_array = np.load('output_array.npy')
null_layer = np.load('null_layer.npy')
# input_array_shape = np.shape(input_array)
# output_array_shape = np.shape(output_array)
# null_layer_shape = np.shape(null_layer)
# distances_shape = moose(len(languages) - 1)
# stabilities_shape = [1, len(features)]
# biases_shape = [1, len(features)]

input_array_shape = [1398628, 195]
output_array_shape = [1398628, 195]
null_layer_shape = [1398628, 195]
distances_shape = [1398628, 1]
stabilities_shape = [1, 195]
biases_shape = [1, 195]


input_placeholder = tf.keras.backend.placeholder(input_array_shape, dtype=tf.float32)
output_placeholder = tf.keras.backend.placeholder(output_array_shape, dtype=tf.float32)
null_layer_placeholder = tf.keras.backend.placeholder(null_layer_shape, dtype=tf.float32)


def moose(x):
  return ((x**2) + x) / 2


class Model(object):
  def __init__(self):
    self.distances = tf.Variable(np.ones(distances_shape), dtype=tf.float32)
    self.stabilities = tf.Variable(np.zeros(stabilities_shape), dtype=tf.float32)
    self.biases = tf.Variable(np.zeros(biases_shape), dtype=tf.float32)
#     self.distances = tf.random.uniform(distances_shape, minval=0, maxval=1, dtype=tf.float32)
#     self.stabilities = tf.random.uniform(stabilities_shape, minval=-3, maxval=-0.01, dtype=tf.float32)
#     self.biases = tf.random.uniform(biases_shape, minval=0, maxval=1, dtype=tf.float32)
 
  def __call__(self, input_placeholder, null_layer_placeholder):
    step1 = tf.matmul(self.distances, self.stabilities) + 3
    step2 = tf.math.sigmoid(step1)
    one = tf.constant(np.array([1]), dtype=tf.float32)
    step3 = tf.math.abs(tf.math.subtract(tf.math.subtract(one, input_placeholder), step2))
    step4 = step3 + self.biases
    step5 = tf.math.divide(step4, (1 + self.biases))
    prediction = tf.math.multiply(step5, null_layer_placeholder)
    return prediction

def loss(predicted_y, target_y):
  return tf.keras.losses.BinaryCrossentropy(predicted_y, target_y)

def train(model, input_array, null_layer, output_array, learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(model(input_array, null_layer), output_array)
  d_distances, d_stabilities, d_biases = t.gradient(current_loss, [model.distances, model.stabilies, model.biases])
  model.distances.assign_sub(learning_rate * d_distances)
  model.stabilities.assign_sub(learning_rate * d_stabilities)
  model.biases.assign_sub(learning_rate * d_biases)


model = Model()

# epochs = range(2)
# 
# for epoch in epochs:
#   print('moose')
#   current_loss = loss(model(input_array, null_layer), output_array)
#   train(model, input_array, null_layer, output_array, learning_rate=0.1)
#   print(current_loss)
  

# 
# steps = 10000
# learn_rate = 0.03
# train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)	
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)


# for i in xrange(steps):
#   feed = {x: X, y_: target}
#   sess.run(train_step, feed_dict = feed)


'''
step1 = (distances * stabilities) + 3

step2 = sigmoid(step1), the probability of staying the same

step3 = abs((1 - input ) - step2), probability of being 1

step4 = step3 + biases

step5 = step4 / (1 + biases), normalising them 

prediction = tf.multiply(step5, null_layer)

loss = product of 1 - abs(output - prediction)
'''



# def tensorflowAnalysis(data):
#     steps = 10000
#     learn_rate = 0.03
# 
# 
#     dataFrame = createMainDataFrame(data, trees, howFarBack, threshold, limit)
#     dataFrame = dataFrame.dropna()
#     target = np.array(dataFrame['TipState'])
#     target = [[member] for member in target]
#     x1 = np.array(dataFrame['AncestorState'])
#     x2 = np.array(dataFrame['NeighbourValue'])
#     X = np.transpose(np.array([x1, x2]))
#     dependent_variables = np.array(X)	
#     x = tf.placeholder(tf.float32, [None, 2], name = "x")
#     W = tf.Variable(tf.zeros([2, 1]), name = "W")
#     b1 = tf.Variable(tf.ones([1]), name = "b1")
#     b2 = tf.Variable(tf.ones([1]), name = "b2")
#     b = b1 * b2
# #     b = tf.Variable(tf.zeros([1]), name = "b")
#     y = tf.matmul(x, W) + b
# #     y = tf.matmul(x, W)
#     y_ = tf.placeholder(tf.float32, [None, 1])
#     cost = tf.reduce_mean(tf.square(y_ - y))
#     cost_sum = tf.summary.scalar("cost", cost)
#     train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)	
#     clip_op = tf.assign(W, tf.clip_by_value(W, 0, np.infty))
#     clip_op2 = tf.assign(b1, tf.clip_by_value(b1, 0, np.infty))
#     clip_op3 = tf.assign(b2, tf.clip_by_value(b2, 0, np.infty))
#     sum = tf.reduce_sum(W) + b1
# #     sum = tf.reduce_sum(W)
#     normalise1 = tf.assign(W, W / sum)
#     normalise2 = tf.assign(b1, b1 / sum)
#     sess = tf.Session()
#     init = tf.initialize_all_variables()
#     sess.run(init)
#     for i in xrange(steps):
#         feed = {x: X, y_: target}
#         sess.run(train_step, feed_dict = feed)
#         sess.run(clip_op)
#         sess.run(clip_op2)
#         sess.run(clip_op3)
# #         print sess.run(b1)
# #         print sess.run(b2)
# #         print sess.run(b)
#         
# #         if sess.run(sum) > 1:
#         sess.run(normalise1)
#         sess.run(normalise2)
#     coefficients = sess.run(W)
#     return str(coefficients[0][0]) + ',' + str(coefficients[1][0]) + ',' + str(sess.run(b1)[0]) + ',' + str(sess.run(b2)[0])


