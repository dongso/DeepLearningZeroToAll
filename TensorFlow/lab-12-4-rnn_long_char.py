from __future__ import print_function
import tensorflow as tf  # tensorflow
import numpy as np  # numpy
from tensorflow.contrib import rnn  # rnn
tf.set_random_seed(777)  # 랜덤 시드 설정 reproducibility
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")  # 샘플 문장
char_set = list(set(sentence))  # 문자 셋
char_dic = {w: i for i, w in enumerate(char_set)}  # 문자와 인덱스 딕셔너리
data_dim = len(char_set)  # 데이터 차수
hidden_size = len(char_set)  # 출력 차수
num_classes = len(char_set)  # 범주
sequence_length = 10  # 시퀀스 길이 Any arbitrary number
learning_rate = 0.1  # 학습율

dataX = []
dataY = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)
    x = [char_dic[c] for c in x_str]  # x 문자를 인덱스로 변환 x str to index
    y = [char_dic[c] for c in y_str]  # y 문자를 인덱스로 변환 y str to index
    dataX.append(x)
    dataY.append(y)
batch_size = len(dataX)
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])
X_one_hot = tf.one_hot(X, num_classes)  # 원핫 인코딩 One-hot encoding
print(X_one_hot)  # shape 확인. check out the shape
# 출력크기를 가진 LSTM 셀 생성. Make a lstm cell with hidden_size (each unit output vector size)
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell
multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)
# outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

# 완전연결 레이어 FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])  # 시퀀스 손실 계산을 위한 reshape. reshape out for sequence_loss
weights = tf.ones([batch_size, sequence_length])  # 모든 가중치는 1. All weights are 1 (equal weights)
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(500):
    _, l, results = sess.run([train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([char_set[t] for t in index]), l)
# 동작하는지 확인하기 위해 각 결과를 출력 Let's print the last char of each result to check it works
results = sess.run(outputs, feed_dict={X: dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # 처음부터 끝까지 문장을 생성한 결과 출력 print all for the first result to make a sentence
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')

'''
0 167 tttttttttt 3.23111
0 168 tttttttttt 3.23111
0 169 tttttttttt 3.23111
…
499 167  of the se 0.229616
499 168 tf the sea 0.229616
499 169   the sea. 0.229616

g you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea.

'''
