"""
In this file, we will implement back propagations by hands

We will use the Sigmoid Cross Entropy loss function.
This is equivalent to tf.nn.sigmoid_softmax_with_logits(logits, labels)

[References]

1) Tensorflow Document (tf.nn.sigmoid_softmax_with_logits)
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

2) Neural Net Backprop in one slide! by Sung Kim
    https://docs.google.com/presentation/d/1_ZmtfEjLmhbuM_PqbDYMXXLAqeWN0HwuhcSKnUQZ6MM/edit#slide=id.g1ec1d04b5a_1_83

3) Back Propagation with Tensorflow by Dan Aloni
    http://blog.aloni.org/posts/backprop-with-tensorflow/

4) Yes you should understand backprop by Andrej Karpathy
    https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.cockptkn7


[Network Architecture]

Input: x
Layer1: x * W + b
Output layer = σ(Layer1)

Loss_i = - y * log(σ(Layer1)) - (1 - y) * log(1 - σ(Layer1))
Loss = tf.reduce_sum(Loss_i)

We want to compute that

dLoss/dW = ???
dLoss/db = ???

please read "Neural Net Backprop in one slide!" for deriving formulas

"""
import tensorflow as tf  # tensorflow
import numpy as np  # numpy
tf.set_random_seed(777)  # 랜덤 시드 설정 for reproducibility
# 다양한 피처에 기초하여 동물의 종류를 예측하기 Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)  # csv 데이터 읽기
X_data = xy[:, 0:-1]  # X 데이터
N = X_data.shape[0]  # N 데이터 갯수
y_data = xy[:, [-1]]  # Y 데이터
# y 데이터는 0~6의 레이블 값을 갖는다 y_data has labels from 0 ~ 6
print("y has one of the following values")
print(np.unique(y_data))
# X_data.shape = (101, 16) => 101 samples, 16 features
# y_data.shape = (101, 1)  => 101 samples, 1 label
print("Shape of X data: ", X_data.shape)
print("Shape of y data: ", y_data.shape)
nb_classes = 7  # 0 ~ 6
X = tf.placeholder(tf.float32, [None, 16])  # X 플레이스홀더
y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6  Y 플레이스홀더
target = tf.one_hot(y, nb_classes)  # one hot 원핫인코딩
target = tf.reshape(target, [-1, nb_classes])  # 출력 클래스에 맞춰 reshape
target = tf.cast(target, tf.float32)  # float32 형변환
W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')  # W 변수
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')  # b 변수

def sigma(x):  # 시그모이드 함수
    # sigmoid function
    # σ(x) = 1 / (1 + exp(-x))
    return 1. / (1. + tf.exp(-x))

def sigma_prime(x):  # 시그모이드 함수 미분
    # derivative of the sigmoid function
    # σ'(x) = σ(x) * (1 - σ(x))
    return sigma(x) * (1. - sigma(x))

# 정방향 전파 Forward propagtion
layer_1 = tf.matmul(X, W) + b
y_pred = sigma(layer_1)

# 손실 함수 (정방향 전파의 마지막) Loss Function (end of forwad propagation)
loss_i = - target * tf.log(y_pred) - (1. - target) * tf.log(1. - y_pred)
loss = tf.reduce_sum(loss_i)

# 차원 확인 Dimension Check
assert y_pred.shape.as_list() == target.shape.as_list()

# 체인룰에 따라 역전파 Back prop (chain rule)
# 어떻게 미분하는가? 슬라이드에서 신경망 역전파 부분을 읽어보세요. How to derive? please read "Neural Net Backprop in one slide!"
d_loss = (y_pred - target) / (y_pred * (1. - y_pred) + 1e-7)
d_sigma = sigma_prime(layer_1)
d_layer = d_loss * d_sigma
d_b = d_layer
d_W = tf.matmul(tf.transpose(X), d_layer)

# 경사값을 사용하여 신경망 업데이트 Updating network using gradients
learning_rate = 0.01  # 학습율
train_step = [
    tf.assign(W, W - learning_rate * d_W),
    tf.assign(b, b - learning_rate * tf.reduce_sum(d_b)),
]
# 예측과 정확도 Prediction and Accuracy
prediction = tf.argmax(y_pred, 1)
acct_mat = tf.equal(tf.argmax(y_pred, 1), tf.argmax(target, 1))
acct_res = tf.reduce_mean(tf.cast(acct_mat, tf.float32))
with tf.Session() as sess: # 그래프 실행 Launch graph
    sess.run(tf.global_variables_initializer())  # 변수 초기화
    for step in range(500):  # 499회 반복
        sess.run(train_step, feed_dict={X: X_data, y: y_data})  # train_step 계산
        if step % 10 == 0:  # 10회 마
            # 300이내에 100% 정확도를 보게된다. Within 300 steps, you should see an accuracy of 100%
            step_loss, acc = sess.run([loss, acct_res], feed_dict={X: X_data, y: y_data})  # loos, acct_res 계산
            print("Step: {:5}\t Loss: {:10.5f}\t Acc: {:.2%}" .format(step, step_loss, acc))  # step, step_loss, acc 출력
    # 예측값 확인한다. Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: X_data})  # prediction 계산
    for p, y in zip(pred, y_data):  # pred, y_data 에서
        msg = "[{}]\t Prediction: {:d}\t True y: {:d}"
        print(msg.format(p == int(y[0]), p, int(y[0])))  # 같은지, p, y 값 출력

"""
Output Example

Step:     0      Loss:  453.74799        Acc: 38.61%
Step:    20      Loss:   95.05664        Acc: 88.12%
Step:    40      Loss:   66.43570        Acc: 93.07%
Step:    60      Loss:   53.09288        Acc: 94.06%
...
Step:   290      Loss:   18.72972        Acc: 100.00%
Step:   300      Loss:   18.24953        Acc: 100.00%
Step:   310      Loss:   17.79592        Acc: 100.00%
...
[True]   Prediction: 0   True y: 0
[True]   Prediction: 0   True y: 0
[True]   Prediction: 3   True y: 3
[True]   Prediction: 0   True y: 0
...
"""

'''
실행결과
Step:   220	 Loss:   23.07096	 Acc: 98.02%
Step:   230	 Loss:   22.32165	 Acc: 98.02%
Step:   240	 Loss:   21.62208	 Acc: 100.00%
...
Step:   470	 Loss:   12.88337	 Acc: 100.00%
Step:   480	 Loss:   12.67214	 Acc: 100.00%
Step:   490	 Loss:   12.46836	 Acc: 100.00%
...
[True]	 Prediction: 0	 True y: 0
[True]	 Prediction: 6	 True y: 6
[True]	 Prediction: 1	 True y: 1
'''
