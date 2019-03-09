# Lab 3 비용 최소화 Minimizing Cost
import tensorflow as tf  # tensorflow
# 텐서플로 그래프 입력 tf Graph Input
X = [1, 2, 3]  # X 데이터
Y = [1, 2, 3]  # Y 데이터
W = tf.Variable(5.0)  # 잘못된 모델 가중치 설정 Set wrong model weights
hypothesis = X * W  # 선형 모델 Linear model
cost = tf.reduce_mean(tf.square(hypothesis - Y))  # 비용/손실 함수 정의 cost/loss function
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)  # 경사 하강 옵티마이저로 최소화 Minimize: Gradient Descent Optimizer
with tf.Session() as sess:  # 세션에서 그래프 실행 Launch the graph in a session.
    sess.run(tf.global_variables_initializer())  # 그래프에서 변수 초기화 Initializes global variables in the graph.
    for step in range(101):  # 100회 반복
        _, W_val = sess.run([train, W])  # train, W 실행
        print(step, W_val)  # step, W 출력
"""
0 5.0
1 1.2666664
2 1.0177778
3 1.0011852
4 1.000079
...
97 1.0
98 1.0
99 1.0
100 1.0
"""
'''
실행결과
0 1.2666664
1 1.0177778
2 1.0011852
3 1.000079
4 1.0000052
5 1.0000004
6 1.0
...
98 1.0
99 1.0
100 1.0
'''
