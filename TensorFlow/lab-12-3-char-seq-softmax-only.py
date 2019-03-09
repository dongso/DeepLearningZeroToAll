# Lab 12 Character Sequence Softmax only
import tensorflow as tf  # tensorflow
import numpy as np  # numpy
tf.set_random_seed(777)  # 랜덤 시드 설정 reproducibility
sample = " if you want you"  # 샘플 문자열
idx2char = list(set(sample))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex
# 하이퍼파라미터값들 hyper parameters
dic_size = len(char2idx)  # RNN 입력크기(원핫 크기). RNN input size (one hot size)
rnn_hidden_size = len(char2idx)  # RNN 출력크기. RNN output size
num_classes = len(char2idx)  # 최종 출력크기(RNN이나 소프트맥스) final output size (RNN or softmax, etc.)
batch_size = 1  # 샘플데이터, 배치크기 one sample data, one batch
sequence_length = len(sample) - 1  # LSTM 펼친 길이(시퀀스 길이) number of lstm rollings (unit #)
learning_rate = 0.1  # 학습율

sample_idx = [char2idx[c] for c in sample]  # char to index
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello
X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label
# 데이터 평탄화 flatten the data (ignore batches for now). No effect if the batch size is 1
X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
X_for_softmax = tf.reshape(X_one_hot, [-1, rnn_hidden_size])
# 소프트맥스 레이어 softmax layer (rnn_hidden_size -> num_classes)
softmax_w = tf.get_variable("softmax_w", [rnn_hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b
# 데이터 사용을 위한 Reshape.  expend the data (revive the batches)
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])
# 시퀀스 비용/손실 계산 Compute sequence cost/loss
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)  # 모든 시퀀스 손실 평균 mean all sequence loss
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
prediction = tf.argmax(outputs, axis=2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})
        result_str = [idx2char[c] for c in np.squeeze(result)]  # 사전을 이용한 문자 출력 print char using dic
        print(i, "loss:", l, "Prediction:", ''.join(result_str))

'''
0 loss: 2.29513 Prediction: yu yny y y oyny
1 loss: 2.10156 Prediction: yu ynu y y oynu
2 loss: 1.92344 Prediction: yu you y u  you

..

2997 loss: 0.277323 Prediction: yf you yant you
2998 loss: 0.277323 Prediction: yf you yant you
2999 loss: 0.277323 Prediction: yf you yant you
'''
