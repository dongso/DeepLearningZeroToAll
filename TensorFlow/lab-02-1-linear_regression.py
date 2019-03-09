# Lab 2 선형 회귀 Linear Regression
import tensorflow as tf  # tensorflow
tf.set_random_seed(777)  # 랜덤 생성 시드값 777로 설정 for reproducibility
x_train = [1, 2, 3]  # 학습용 X 데이터
y_train = [1, 2, 3]  # 학습용 Y 데이터
# y_data = x_data * W + b를 계산하는 것으로 (가중치) W와 (바이어스) b 값을 구하려 한다. Try to find values for W and b to compute y_data = x_data * W + b
# W는 1이고, b는 0이라는 것을 알고 있다. We know that W should be 1 and b should be 0
# 하지만 텐서플로가 찾아내도록 해본다. But let TensorFlow figure it out
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")
hypothesis = x_train * W + b  # 가설은 XW+b 이다. Our hypothesis XW+b
cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # 비용/손실 함수를 정의한다. cost/loss function
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)  # (경사 하강법을 자동으로 계산해주는) 옵티마이저를 이용한다. optimizer
with tf.Session() as sess:  # 세션에서 그래프를 실행한다. Launch the graph in a session.
    sess.run(tf.global_variables_initializer())  # 그래프에서 변수들을 초기화 한다. Initializes global variables in the graph.
    for step in range(2001):  # 잘 맞는 선을 찾는다. Fit the line
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])  # train, cost, W, b에 대한 결과를 얻도록 작업을 실행한다.
        if step % 20 == 0:  # 20 step 마다 (나누어 떨어짐. 나머지가 0임)
            print(step, cost_val, W_val, b_val)  # 현재 step 에서 구한 결과인 step, cost, W, b를 출력한다.
# 학습 결과 가장 적합한 값은 W:[1], b:[0]. Learns best fit W:[ 1.],  b:[ 0.]
"""
0 2.82329 [ 2.12867713] [-0.85235667]
20 0.190351 [ 1.53392804] [-1.05059612]
40 0.151357 [ 1.45725465] [-1.02391243]
...
1960 1.46397e-05 [ 1.004444] [-0.01010205]
1980 1.32962e-05 [ 1.00423515] [-0.00962736]
2000 1.20761e-05 [ 1.00403607] [-0.00917497]
"""
'''
실행결과
1960 1.4711059e-05 [1.004444] [-0.01010205]
1980 1.3360998e-05 [1.0042351] [-0.00962736]
2000 1.21343355e-05 [1.0040361] [-0.00917497]
'''