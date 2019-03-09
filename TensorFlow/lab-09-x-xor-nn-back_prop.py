# Lab 9 XOR-back_prop
import tensorflow as tf  # tensorflow
import numpy as np  # numpy
tf.set_random_seed(777)  # 랜덤 시드 설정 for reproducibility
learning_rate = 0.1  # 학습율
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]  # x 데이터
y_data = [[0], [1], [1], [0]]  # y 데이터
x_data = np.array(x_data, dtype=np.float32)  # x 데이터 np 배열
y_data = np.array(y_data, dtype=np.float32)  # y 데이터 np 배열
X = tf.placeholder(tf.float32, [None, 2])  # X 플레이스홀더
Y = tf.placeholder(tf.float32, [None, 1])  # Y 플레이스홀더
W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')  # W1 변수
b1 = tf.Variable(tf.random_normal([2]), name='bias1')  # b1 변수
l1 = tf.sigmoid(tf.matmul(X, W1) + b1)  # layer1
W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')  # W2 변수
b2 = tf.Variable(tf.random_normal([1]), name='bias2')  # b2 변수
Y_pred = tf.sigmoid(tf.matmul(l1, W2) + b2)  # 가설 Y_pred
cost = -tf.reduce_mean(Y * tf.log(Y_pred) + (1 - Y) * tf.log(1 - Y_pred))  # 비용/손실 함수 cost/loss function
# 신경망 Network
#          p1     a1           l1     p2     a2           l2 (y_pred)
# X -> (*) -> (+) -> (sigmoid) -> (*) -> (+) -> (sigmoid) -> (loss)
#       ^      ^                   ^      ^
#       |      |                   |      |
#       W1     b1                  W2     b2

d_Y_pred = (Y_pred - Y) / (Y_pred * (1.0 - Y_pred) + 1e-7)  # 손실 함수 미분 Loss derivative

# 레이어2 Layer 2
d_sigma2 = Y_pred * (1 - Y_pred)
d_a2 = d_Y_pred * d_sigma2
d_p2 = d_a2
d_b2 = d_a2
d_W2 = tf.matmul(tf.transpose(l1), d_p2)

# 평균 Mean
d_b2_mean = tf.reduce_mean(d_b2, axis=[0])
d_W2_mean = d_W2 / tf.cast(tf.shape(l1)[0], dtype=tf.float32)

# 레이어1 Layer 1
d_l1 = tf.matmul(d_p2, tf.transpose(W2))
d_sigma1 = l1 * (1 - l1)
d_a1 = d_l1 * d_sigma1
d_b1 = d_a1
d_p1 = d_a1
d_W1 = tf.matmul(tf.transpose(X), d_a1)

# 평균 Mean
d_W1_mean = d_W1 / tf.cast(tf.shape(X)[0], dtype=tf.float32)
d_b1_mean = tf.reduce_mean(d_b1, axis=[0])

# 가중치 업데이트 Weight update
step = [
  tf.assign(W2, W2 - learning_rate * d_W2_mean),
  tf.assign(b2, b2 - learning_rate * d_b2_mean),
  tf.assign(W1, W1 - learning_rate * d_W1_mean),
  tf.assign(b1, b1 - learning_rate * d_b1_mean)
]

# 정확도 계산 Accuracy computation
# 가설이 0.5보다 크면 참이고, 아니면 거짓 True if hypothesis > 0.5 else False
predicted = tf.cast(Y_pred > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:  # 그래프 실행 Launch graph
    sess.run(tf.global_variables_initializer())  # 변수 초기화 Initialize TensorFlow variables
    print("shape", sess.run(tf.shape(X)[0], feed_dict={X: x_data}))  # X 데이터 [0] shape
    for i in range(10001):  # 10000회 반복
        sess.run([step, cost], feed_dict={X: x_data, Y: y_data})  # step, cost 실행
        if i % 1000 == 0:  # 1000회 마다
            print(i, sess.run([cost, d_W1], feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2]))  # i번째, cost, d_W1 계산
    # 정확도 출력 Accuracy report
    h, c, a = sess.run([Y_pred, predicted, accuracy], feed_dict={X: x_data, Y: y_data})  #   Y_pred, p, a 계산
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)  # h, c, a 출력

'''
Hypothesis:  [[ 0.01338224]
 [ 0.98166382]
 [ 0.98809403]
 [ 0.01135806]]
Correct:  [[ 0.]
 [ 1.]
 [ 1.]
 [ 0.]]
Accuracy:  1.0
'''

'''
실행결과
9000 [0.01614568, array([[-0.00347966, -0.00493037],
       [ 0.00372719,  0.00505056]], dtype=float32)] [array([[ 6.1832714,  6.0115848],
       [-6.301682 , -5.7031956]], dtype=float32), array([[ 9.818268 ],
       [-9.3095255]], dtype=float32)]
10000 [0.013844881, array([[-0.00296173, -0.00414468],
       [ 0.0031764 ,  0.00424056]], dtype=float32)] [array([[ 6.2634654,  6.1245055],
       [-6.3876405, -5.8188014]], dtype=float32), array([[10.100027],
       [-9.598649]], dtype=float32)]

Hypothesis:  [[0.01338229]
 [0.9816638 ]
 [0.98809403]
 [0.01135809]] 
Correct:  [[0.]
 [1.]
 [1.]
 [0.]] 
Accuracy:  1.0
'''