# Lab 3 비용 최소화 Minimizing Cost
import tensorflow as tf  # tensorflow
import matplotlib.pyplot as plt  # matplotlib.pyplot
X = [1, 2, 3]  # X 데이터
Y = [1, 2, 3]  # Y 데이터
W = tf.placeholder(tf.float32)  # W 플레이스홀더
hypothesis = X * W  # 리니어 모델 X * W에 대한 가설 h 정의. Our hypothesis for linear model X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))  # 비용/손실 함수 정의한다. cost/loss function
# 비용함수를 그래프 출력하기 위한 변수들. Variables for plotting cost function
W_history = []  # W 값을 저장할 리스트
cost_history = []  # cost 값을 저장할 리스트
with tf.Session() as sess:  # 세션에서 그래프를 실행한다. Launch the graph in a session.
    for i in range(-30, 50):  # -30~50 구간 동안 반복
        curr_W = i * 0.1  # 실제로는 -3.0~5.0 구간 값으로 W 설정
        curr_cost = sess.run(cost, feed_dict={W: curr_W})  # curr_W를 입력 데이터로 cost에 대한 작업을 실행 
        W_history.append(curr_W)  # W 저장
        cost_history.append(curr_cost)  # cost 저장
plt.plot(W_history, cost_history)  # 비용 함수 결과를 화면 출력한다. Show the cost function
plt.show()
