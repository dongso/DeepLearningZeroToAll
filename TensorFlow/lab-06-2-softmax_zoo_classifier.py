# Lab 6 소프트맥스 분류기 Softmax Classifier
import tensorflow as tf  # tensorflow
import numpy as np  # numpy
tf.set_random_seed(777)  # 랜덤 시드 생성 for reproducibility
# 다양한 피처에 기초한 동물 종류 예측하기 Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)  # numpy로 csv 데이터 읽기
x_data = xy[:, 0:-1]  # 마지막 열 전까지 X
y_data = xy[:, [-1]]  # 마지막 열만 Y
print(x_data.shape, y_data.shape)  # x, y 데이터 shape 출력
'''
(101, 16) (101, 1)
'''
nb_classes = 7  # 0 ~ 6
X = tf.placeholder(tf.float32, [None, 16])  # X 플레이스홀더
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6  Y 플레이스홀더
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot  원핫 인코딩
print("one_hot:", Y_one_hot)  # Y_one_hot 출력
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])  #  nb_classes 수를 기준으로 reshape 하기
print("reshape one_hot:", Y_one_hot)  # Y_one_hot 출력
'''
one_hot: Tensor("one_hot:0", shape=(?, 1, 7), dtype=float32)
reshape one_hot: Tensor("Reshape:0", shape=(?, 7), dtype=float32)
'''
W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')  # W 변수
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')  # b 변수
# 소프트맥스 함수 계산 tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.stop_gradient([Y_one_hot])))  # 크로스 엔트로피 비용/손실 함수 Cross entropy cost/loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)  # 경사하강법 비용 최소화
prediction = tf.argmax(hypothesis, 1)  # 예측 결과는 h 결과에서 가장 큰 값의 번호
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))  # 예측 결과와 정답을 비교하여 같으면 성공
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 성공한 정확도 계산
with tf.Session() as sess:  # 그래프 실행 Launch graph
    sess.run(tf.global_variables_initializer())  # 변수 초기화
    for step in range(2001):  # 2000회 반복
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: x_data, Y: y_data})  # optimizer, cost, accuracy 계산        
        if step % 100 == 0:  # 100회 마다
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))  # step, cost, acc 출력
    pred = sess.run(prediction, feed_dict={X: x_data})  # 예측이 잘되는지 확인 Let's see if we can predict
    for p, y in zip(pred, y_data.flatten()):  # pred와 y flatten을 zip 하기 y_data: (N,1) = flatten => (N, ) matches pred.shape
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))  # 예측값과 y 가 같은지, 예측값, y값 출력
'''
Step:     0 Loss: 5.106 Acc: 37.62%
Step:   100 Loss: 0.800 Acc: 79.21%
Step:   200 Loss: 0.486 Acc: 88.12%
...
Step:  1800	Loss: 0.060	Acc: 100.00%
Step:  1900	Loss: 0.057	Acc: 100.00%
Step:  2000	Loss: 0.054	Acc: 100.00%
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 3 True Y: 3
...
[True] Prediction: 0 True Y: 0
[True] Prediction: 6 True Y: 6
[True] Prediction: 1 True Y: 1
'''

'''
실행결과
Step:  1800	Cost: 0.060	Acc: 100.00%
Step:  1900	Cost: 0.057	Acc: 100.00%
Step:  2000	Cost: 0.054	Acc: 100.00%
...
[True] Prediction: 0 True Y: 0
[True] Prediction: 6 True Y: 6
[True] Prediction: 1 True Y: 1
'''
