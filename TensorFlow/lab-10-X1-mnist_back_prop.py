# http://blog.aloni.org/posts/backprop-with-tensorflow/
# https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.b3rvzhx89
import tensorflow as tf  # tensofflow
tf.set_random_seed(777)  # 랜덤 시드 설정 reproducibility
# Check out https://www.tensorflow.org/get_started/mnist/beginners for more information about the mnist dataset
from tensorflow.examples.tutorials.mnist import input_data  # tensorflow mnist
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # mnist
X = tf.placeholder(tf.float32, [None, 784])  # X 플레이스홀더
Y = tf.placeholder(tf.float32, [None, 10])  # Y 플레이스홀더
w1 = tf.Variable(tf.truncated_normal([784, 30]))  # w1 변수
b1 = tf.Variable(tf.truncated_normal([1, 30]))  # b1 변수
w2 = tf.Variable(tf.truncated_normal([30, 10]))  # w2 변수
b2 = tf.Variable(tf.truncated_normal([1, 10]))  # b2 변수
def sigma(x):  #  시그모이드 함수 sigmoid function
    return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(-x)))

def sigma_prime(x):  # 시그모이드 함수의 미분 derivative of the sigmoid function
    return sigma(x) * (1 - sigma(x))

# 정방향 전파 Forward prop
l1 = tf.add(tf.matmul(X, w1), b1)
a1 = sigma(l1)
l2 = tf.add(tf.matmul(a1, w2), b2)
y_pred = sigma(l2)

# diff
assert y_pred.shape.as_list() == Y.shape.as_list()
diff = (y_pred - Y)

# 체인룰에 따른 역전파 Back prop (chain rule)
d_l2 = diff * sigma_prime(l2)
d_b2 = d_l2
d_w2 = tf.matmul(tf.transpose(a1), d_l2)
d_a1 = tf.matmul(d_l2, tf.transpose(w2))
d_l1 = d_a1 * sigma_prime(l1)
d_b1 = d_l1
d_w1 = tf.matmul(tf.transpose(X), d_l1)

# 경사값을 이용한 신경망 업데이트 Updating network using gradients
learning_rate = 0.5  # 학습율
step = [
    tf.assign(w1, w1 - learning_rate * d_w1),
    tf.assign(b1, b1 - learning_rate * tf.reduce_mean(d_b1, reduction_indices=[0])),
    tf.assign(w2, w2 - learning_rate * d_w2),
    tf.assign(b2, b2 - learning_rate * tf.reduce_mean(d_b2, reduction_indices=[0]))
]

# 훈련 과정 실행 및 검증. 7. Running and testing the training process
acct_mat = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(10000):  # 9999회 반복
    batch_xs, batch_ys = mnist.train.next_batch(10)  # 배치 데이터
    sess.run(step, feed_dict={X: batch_xs, Y: batch_ys})  # step 계산
    if i % 1000 == 0:  # 1000회 마다
        res = sess.run(acct_res, feed_dict={X: mnist.test.images[:1000], Y: mnist.test.labels[:1000]})  # acct_res 계산
        print(res)  # res 출력

# 텐서플로우 자동 미분. 8. Automatic differentiation in TensorFlow
cost = diff * diff
step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

'''
실행결과
116.0
840.0
878.0
873.0
897.0
892.0
904.0
916.0
908.0
919.0
'''