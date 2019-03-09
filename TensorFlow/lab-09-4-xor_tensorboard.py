# Lab 9 XOR
import tensorflow as tf  # tensorflow
import numpy as np  # numpy
tf.set_random_seed(777)  # 랜덤 시드 설정 for reproducibility
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)  # x 데이터
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)  # y 데이터
X = tf.placeholder(tf.float32, [None, 2], name="x")  # X 플레이스홀더
Y = tf.placeholder(tf.float32, [None, 1], name="y")  # Y 플레이스홀더
with tf.name_scope("Layer1"):  # Layer1 scope 설정
    W1 = tf.Variable(tf.random_normal([2, 2]), name="weight_1")  # W1 변수
    b1 = tf.Variable(tf.random_normal([2]), name="bias_1")  # b1 변수
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)  # layer1
    tf.summary.histogram("W1", W1)  # W1을 summary 히스토그램 저장
    tf.summary.histogram("b1", b1)  # b1을 summary 히스토그램 저장
    tf.summary.histogram("Layer1", layer1)  # layer1을 summary 히스토그램 저장

with tf.name_scope("Layer2"):  # Layer2 scope 설정
    W2 = tf.Variable(tf.random_normal([2, 1]), name="weight_2")  # W2 변수
    b2 = tf.Variable(tf.random_normal([1]), name="bias_2")  # b2 변수
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)  # layer2
    tf.summary.histogram("W2", W2)  # W2을 summary 히스토그램 저장
    tf.summary.histogram("b2", b2)  # b2을 summary 히스토그램 저장
    tf.summary.histogram("Hypothesis", hypothesis)  # 가설 h를 summary 히스토그램 저장

# 비용/손실 함수 cost/loss function
with tf.name_scope("Cost"):  # Cost scope 설정
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))  # 비용 함수
    tf.summary.scalar("Cost", cost)  # cost를 summary 스칼라 저장

with tf.name_scope("Train"):  # Train scope 설정
    train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)  # 경사하강법 비용 최소화

# 정확도 계산 Accuracy computation
# 가설이 0.5보다 크면 참이고, 아니면 거짓 True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
tf.summary.scalar("accuracy", accuracy)  # 정확도를 summary 스칼라 저장
with tf.Session() as sess:  # 그래프 실행 Launch graph
    # tensorboard --logdir=./logs/xor_logs
    merged_summary = tf.summary.merge_all()  # summary 저장값 병합
    writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")  # 로그 파일로 저장
    writer.add_graph(sess.graph)  # 세션 그래프 추가 Show the graph
    sess.run(tf.global_variables_initializer())  # 변수 초기화 Initialize TensorFlow variables
    for step in range(10001):  # 10000회 반복
        _, summary, cost_val = sess.run([train, merged_summary, cost], feed_dict={X: x_data, Y: y_data})  # train, merged_summary, cost 계산
        writer.add_summary(summary, global_step=step)  # step별 summary 값 추가
        if step % 100 == 0:  # 100회 바다
            print(step, cost_val)  # step, cost 출력
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})  # 정확도 출력 Accuracy report
    print(f"\nHypothesis:\n{h} \nPredicted:\n{p} \nAccuracy:\n{a}")  # h, p, a 출력
"""
Hypothesis:
[[6.1310326e-05]
 [9.9993694e-01]
 [9.9995077e-01]
 [5.9751470e-05]] 
Predicted:
[[0.]
 [1.]
 [1.]
 [0.]] 
Accuracy:
1.0
"""

'''
실행결과
9800 6.4539025e-05
9900 6.136488e-05
10000 5.8384467e-05

Hypothesis:
[[6.1310326e-05]
 [9.9993694e-01]
 [9.9995077e-01]
 [5.9751470e-05]] 
Predicted:
[[0.]
 [1.]
 [1.]
 [0.]] 
Accuracy:
1.0
'''