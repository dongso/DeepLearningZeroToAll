# Lab 2 선형 회귀 Linear Regression
import tensorflow as tf  # tensorflow
tf.set_random_seed(777)  # 랜덤 생성 시드값 777로 설정 for reproducibility
# Y = W * X + b를 계산하는 것으로 (가중치) W와 (바이어스) b 값을 구하려 한다. Try to find values for W and b to compute Y = W * X + b
W = tf.Variable(tf.random_normal([1]), name="weight")  # weight 변수 선언. 초기값은 normal 분포의 random 값으로 W에 저장
b = tf.Variable(tf.random_normal([1]), name="bias")  # bias 변수 선언. 초기값은 normal 분포의 random 값으로 b에 저장
# 텐서를 계산하는데 사용할 플레이스홀더를 선언한다. 이 값은 추구 feed_dict로 입력한다. placeholders for a tensor that will be always fed using feed_dict
# See http://stackoverflow.com/questions/36693740/
X = tf.placeholder(tf.float32, shape=[None])  # float32 형식의 플레이스홀더 X
Y = tf.placeholder(tf.float32, shape=[None])  # float32 형식의 플레이스홀더 Y
hypothesis = X * W + b  # 가설은 X * W + b 이다. Our hypothesis is X * W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))  # 비용/손실 함수를 정의한다. cost/loss function
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)  # 옵티마이저를 사용한다. optimizer
with tf.Session() as sess:  # 세션에서 그래프를 실행한다. Launch the graph in a session.
    sess.run(tf.global_variables_initializer())  # 그래프에서 변수들을 초기화한다. Initializes global variables in the graph.
    for step in range(2001):  # 잘 맞는 선을 찾는다. Fit the line
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})  # train, cost, W, b에 대한 결과를 얻도록 작업을 실행한다.
        if step % 20 == 0:  # 20 step 마다 (나누어 떨어짐. 나머지가 0임)
            print(step, cost_val, W_val, b_val)  # 현재 step 에서 구한 결과인 step, cost, W, b를 출력한다.
    # 모델을 테스트 한다. Testing our model
    print(sess.run(hypothesis, feed_dict={X: [5]}))  # X값 5를 입력한 결과를 출력한다.
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))  # X값 2.5를 입력한 결과를 출력한다.
    print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))  # X값 [1.5, 3.5]를 입력한 결과를 출력한다.
    # 학습 결과 가장 적합한 값은 W:[1], b:[0]. Learns best fit W:[ 1.],  b:[ 0.]
    """
    0 3.5240757 [2.2086694] [-0.8204183]
    20 0.19749963 [1.5425726] [-1.0498911]
    ...
    1980 1.3360998e-05 [1.0042454] [-0.00965055]
    2000 1.21343355e-05 [1.0040458] [-0.00919707]
    [5.0110054]
    [2.500915]
    [1.4968792 3.5049512]
    """
    for step in range(2001):  # 새로운 훈련 데이터로 잘 맞는 선을 찾는다. Fit the line with new training data
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]},) # train, cost, W, b에 대한 결과를 얻도록 작업을 실행한다.
        if step % 20 == 0:  # 20 step 마다 (나누어 떨어짐. 나머지가 0임)
            print(step, cost_val, W_val, b_val)  # 현재 step 에서 구한 결과인 step, cost, W, b를 출력한다.
    # 모델을 테스트 한다. Testing our model
    print(sess.run(hypothesis, feed_dict={X: [5]}))  # X값 5를 입력한 결과를 출력한다.
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))  # X값 2.5를 입력한 결과를 출력한다.
    print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))  # X값 [1.5, 3.5]를 입력한 결과를 출력한다.
    # 학습 결과 가장 적합한 값은 W:[1], b:[1.1]. Learns best fit W:[ 1.],  b:[ 1.1]
    """
    0 1.2035878 [1.0040361] [-0.00917497]
    20 0.16904518 [1.2656431] [0.13599995]
    ...
    1980 2.9042917e-07 [1.00035] [1.0987366]
    2000 2.5372992e-07 [1.0003271] [1.0988194]
    [6.1004534]
    [3.5996385]
    [2.5993123 4.599964 ]
    """
'''
실행결과
1960 3.3239553e-07 [1.000373] [1.098653]
1980 2.9042917e-07 [1.0003488] [1.0987409]
2000 2.5372992e-07 [1.000326] [1.0988233]
[6.1004534]
[3.5996385]
[2.5993123 4.599964 ]
'''