# Lab 3 비용 최소화 Minimizing Cost
import tensorflow as tf  # tensorflow
tf.set_random_seed(777)  # 랜덤 생성 시드값 777로 설정 for reproducibility
x_data = [1, 2, 3]  # X 데이터
y_data = [1, 2, 3]  # Y 데이터
# y_data = W * x_data를 계산하는 것으로 (가중치) W와 (바이어스) b 값을 구하려 한다. Try to find values for W and b to compute y_data = W * x_data
# W는 1값이어야 한다. We know that W should be 1
# 하지만 텐서플로우가 찾도록 해본다. But let's use TensorFlow to figure it out
W = tf.Variable(tf.random_normal([1]), name="weight")  # weight 변수 선언. normal 분포의 random 값 W
X = tf.placeholder(tf.float32)  # X 플레이스홀더
Y = tf.placeholder(tf.float32)  # Y 플레이스홀더
hypothesis = X * W  # 리니어 모델 X * W에 대한 가설 h 정의. Our hypothesis for linear model X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))  # 비용/손실 함수 정의한다. cost/loss function
# 미분을 이용한 경사하강법으로 최소화 한다. Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
learning_rate = 0.1  # 학습율 0.1 설정
gradient = tf.reduce_mean((W * X - Y) * X)  # 경사 값은 비용함수를 미분한 수식으로 정의
descent = W - learning_rate * gradient  # W에서 학습율 * 경사값 만큼 하강한다.
update = W.assign(descent)  # 새로운 W에 값을 update 변수에 할당한다.
with tf.Session() as sess:  # 세션에서 그래프를 실행한다. Launch the graph in a session.
    sess.run(tf.global_variables_initializer())  # 그래프에서 변수를 초기화한다. Initializes global variables in the graph.
    for step in range(21):  # 20회 반복
        _, cost_val, W_val = sess.run([update, cost, W], feed_dict={X: x_data, Y: y_data})  # update, cost, W 를 계산한다.
        print(step, cost_val, W_val)  # step, cost, W 값을 출력한다.
"""
0 1.93919 [ 1.64462376]
1 0.551591 [ 1.34379935]
2 0.156897 [ 1.18335962]
3 0.0446285 [ 1.09779179]
4 0.0126943 [ 1.05215561]
5 0.00361082 [ 1.0278163]
6 0.00102708 [ 1.01483536]
7 0.000292144 [ 1.00791216]
8 8.30968e-05 [ 1.00421977]
9 2.36361e-05 [ 1.00225055]
10 6.72385e-06 [ 1.00120032]
11 1.91239e-06 [ 1.00064015]
12 5.43968e-07 [ 1.00034142]
13 1.54591e-07 [ 1.00018203]
14 4.39416e-08 [ 1.00009704]
15 1.24913e-08 [ 1.00005174]
16 3.5322e-09 [ 1.00002754]
17 9.99824e-10 [ 1.00001466]
18 2.88878e-10 [ 1.00000787]
19 8.02487e-11 [ 1.00000417]
20 2.34053e-11 [ 1.00000226]
"""
'''
실행결과
18 9.998237e-10 [1.0000079]
19 2.8887825e-10 [1.0000042]
20 8.02487e-11 [1.0000023]
'''
