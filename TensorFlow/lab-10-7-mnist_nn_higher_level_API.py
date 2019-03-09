# Lab 10 MNIST와 고수준 텐서플로 API. MNIST and High-level TF API
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout  # fully_connected, batch_norm, dropout
from tensorflow.contrib.framework import arg_scope  # arg_scope
import tensorflow as tf  # tensorflow
import random  # random
# import matplotlib.pyplot as plt  # matplotlib.pyplot
from tensorflow.examples.tutorials.mnist import input_data  # tensorflow mnist
tf.set_random_seed(777)  # 랜덤 시드 설정 reproducibility
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # mnist
# Check out https://www.tensorflow.org/get_started/mnist/beginners for more information about the mnist dataset
# 파라미터값들 parameters
learning_rate = 0.01  # 배치놈을 사용하여 큰 학습율을 설정할 수 있다. we can use large learning rate using Batch Normalization
training_epochs = 15  # 에폭 횟수
batch_size = 100  # 배치 크기
keep_prob = 0.7  # 드롭아웃 비율
# 입력 플레이스홀더 input place holders
X = tf.placeholder(tf.float32, [None, 784])  # X 플레이스홀더
Y = tf.placeholder(tf.float32, [None, 10])  # Y 플레이스홀더
train_mode = tf.placeholder(tf.bool, name='train_mode')  # train_mode 플레이스홀더
# 레이어 출력 크기 layer output size
hidden_output_size = 512  # 히든 출력 크기
final_output_size = 10  # 최종 출력 크기
xavier_init = tf.contrib.layers.xavier_initializer()  # xaiver 초기화
bn_params = {
    'is_training': train_mode,
    'decay': 0.9,
    'updates_collections': None
}
# 'arg_scope'을 사용해서 중복 작성을 줄임. We can build short code using 'arg_scope' to avoid duplicate code same function with different arguments
with arg_scope([fully_connected],
               activation_fn=tf.nn.relu,
               weights_initializer=xavier_init,
               biases_initializer=None,
               normalizer_fn=batch_norm,
               normalizer_params=bn_params
               ):
    hidden_layer1 = fully_connected(X, hidden_output_size, scope="h1")  # h1
    h1_drop = dropout(hidden_layer1, keep_prob, is_training=train_mode)  # h1 drop
    hidden_layer2 = fully_connected(h1_drop, hidden_output_size, scope="h2")  # h2
    h2_drop = dropout(hidden_layer2, keep_prob, is_training=train_mode)  # h2 drop
    hidden_layer3 = fully_connected(h2_drop, hidden_output_size, scope="h3")  # h3
    h3_drop = dropout(hidden_layer3, keep_prob, is_training=train_mode)  # h3 drop
    hidden_layer4 = fully_connected(h3_drop, hidden_output_size, scope="h4")  # h4
    h4_drop = dropout(hidden_layer4, keep_prob, is_training=train_mode)  # h4 drop
    hypothesis = fully_connected(h4_drop, final_output_size, activation_fn=None, scope="hypothesis")  # 가설 h

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))  # 비용/손실/옵티마이저 define cost/loss & optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # 비용 최소화
# 초기화 initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 모델 훈련 train my model
for epoch in range(training_epochs):  # 에폭 횟수
    avg_cost = 0  # 평균 비용
    total_batch = int(mnist.train.num_examples / batch_size)  # 배치 횟수
    for i in range(total_batch):  # 배치 반복
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 배치 데이터
        feed_dict_train = {X: batch_xs, Y: batch_ys, train_mode: True}  # 학습용 feed_dict
        feed_dict_cost = {X: batch_xs, Y: batch_ys, train_mode: False}  # 검증용 feed_dict
        opt = sess.run(optimizer, feed_dict=feed_dict_train)  # optimizer 계산
        c = sess.run(cost, feed_dict=feed_dict_cost)  # cost 계산
        avg_cost += c / total_batch  # 평균 비용 계산
    print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_cost))  # 에폭, 평균 비용
    #print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning Finished!')  # 학습 완료
# 모델 검증 및 정확도 확인 Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, train_mode: False})) # 정확도
# 랜덤 데이터로 예측 Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], train_mode: False}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()

'''
[Epoch:    1] cost = 0.519417209
[Epoch:    2] cost = 0.432551052
[Epoch:    3] cost = 0.404978843
[Epoch:    4] cost = 0.392039919
[Epoch:    5] cost = 0.382165317
[Epoch:    6] cost = 0.377987834
[Epoch:    7] cost = 0.372577601
[Epoch:    8] cost = 0.367208552
[Epoch:    9] cost = 0.365525589
[Epoch:   10] cost = 0.361964276
[Epoch:   11] cost = 0.359540287
[Epoch:   12] cost = 0.356423751
[Epoch:   13] cost = 0.354478216
[Epoch:   14] cost = 0.353212552
[Epoch:   15] cost = 0.35230893
Learning Finished!
Accuracy: 0.9826
'''

'''
실행결과
[Epoch:   13] cost = 0.302896606
[Epoch:   14] cost = 0.301597439
[Epoch:   15] cost = 0.302011495
Learning Finished!
Accuracy: 0.9861
Label:  [4]
Prediction:  [4]
'''