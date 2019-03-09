# From https://www.tensorflow.org/get_started/get_started
import tensorflow as tf  # tensorflow
# 훈련 데이터 training data
x_train = [1, 2, 3, 4]  # X 훈련 데이터
y_train = [0, -1, -2, -3]  # Y 훈련 데이터
# 모델 파라미터값들 Model parameters
W = tf.Variable([0.3], tf.float32)  # 가중치 W 변수
b = tf.Variable([-0.3], tf.float32)  # 바이어스 b 변수
# 모델 입력과 출력 Model input and output
x = tf.placeholder(tf.float32)  # x 플레이스홀더
y = tf.placeholder(tf.float32)  # y 플레이스홀더
hypothesis = x * W + b  # 가설 h = x * W + b
# 비용/손실 함수 정의 cost/loss function
cost = tf.reduce_sum(tf.square(hypothesis - y))  # 에러 제곱의 합을 cost로 정의한다. sum of the squares
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)  # 옵티마이저를 사용한다. optimizer
with tf.Session() as sess:  # 훈련한다. training
    sess.run(tf.global_variables_initializer())  # 변수 초기화 한다.
    for step in range(1000):  # 1000번 반복한다.
        sess.run(train, {x: x_train, y: y_train})  # x_train, y_train 훈련 데이터로 입력하고, train에 대한 작업을 실행한다.
    # 훈련 정확도를 평가한다. evaluate training accuracy
    W_val, b_val, cost_val = sess.run([W, b, cost], feed_dict={x: x_train, y: y_train})  # x_train, y_train 훈련 데이터로 W, b, cost에 대한 작업을 실행한다.
    print(f"W: {W_val} b: {b_val} cost: {cost_val}")  # W, b, cost 값을 출력한다.
"""
W: [-0.9999969] b: [0.9999908] cost: 5.699973826267524e-11
"""
'''
실행결과
W: [-0.9999969] b: [0.9999908] cost: 5.699973826267524e-11
'''
