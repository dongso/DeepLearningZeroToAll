# Lab 9 XOR
import tensorflow as tf  # tensorflow
import numpy as np  # numpy
tf.set_random_seed(777)  # 랜덤 시드 설정 for reproducibility
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)  # x 데이터
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)  # y 데이터
X = tf.placeholder(tf.float32, [None, 2])  # X 플레이스홀더
Y = tf.placeholder(tf.float32, [None, 1])  # Y 플레이스홀더
W = tf.Variable(tf.random_normal([2, 1]), name="weight")  # W 변수
b = tf.Variable(tf.random_normal([1]), name="bias")  # b 변수
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)  # 시그모이스 사용한 가설 Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))  # 비용/손실 함수 cost/loss function
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)  # 경사하강법 비용 최소화
# 정확도 계산 Accuracy computation
# 0.5보다 크면 참, 아니면 거짓 True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
with tf.Session() as sess:  # 그래프 실행 Launch graph
    sess.run(tf.global_variables_initializer())  # 변수 초기화 Initialize TensorFlow variables
    for step in range(10001):  # 10000회 반복
        _, cost_val, w_val = sess.run([train, cost, W], feed_dict={X: x_data, Y: y_data})  # train, cost, W 실행
        if step % 100 == 0:  # 100회 마다
            print(step, cost_val, w_val)  # step, cost, w 출력4
    # 정확도 출력 Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})  # h, p, a 계산
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)  # h, c, a 출력
'''
Hypothesis:  [[ 0.5]
 [ 0.5]
 [ 0.5]
 [ 0.5]]
Correct:  [[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]
Accuracy:  0.5
'''

'''
실행결과
9800 0.6931472 [[1.2228770e-07]
 [1.2214579e-07]]
9900 0.6931472 [[1.2228770e-07]
 [1.2214579e-07]]
10000 0.6931472 [[1.2228770e-07]
 [1.2214579e-07]]

Hypothesis:  [[0.5]
 [0.5]
 [0.5]
 [0.5]] 
Correct:  [[0.]
 [0.]
 [0.]
 [0.]] 
Accuracy:  0.5
'''
