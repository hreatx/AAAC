import numpy as np
import tensorflow as tf
import multiprocessing
import threading
import Queue
import random
import cv2
import gym
import os
from collections import deque

gym.envs.register(id='bo-v0', entry_point='gym.envs.atari:AtariEnv',
                  kwargs={'game': 'breakout', 'obs_type': 'image', 'frameskip': 4, 'repeat_action_probability': 0.0},
                  max_episode_steps=100000,
                  nondeterministic=False, )
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

###############################constant###################################
NUM_OF_ACTION = 4
T = 0
replace_freq = 40000
TRAIN_EPISODE = 100000
NUM_OF_WORKERS = 4  # multiprocessing.cpu_count()
LITTLE_CONST = 1e-7


###############################constant###################################

class Syn:
    def __init__(self, id):
        self.id = id


q_ = [Queue.Queue() for n in range(NUM_OF_WORKERS)]

q = Queue.Queue()


def clipped_reward(r):
    return max(-1, min(r, 1))


# def getEpilson():
#     p = [0.4, 0.3, 0.3]
#     ep = [0.1, 0.01, 0.5]
#     return np.random.choice(ep, 1, p=p)

def preprocess(raw):
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    raw = cv2.resize(raw, (84, 84))
    ret, raw = cv2.threshold(raw, 1, 255, cv2.THRESH_BINARY)
    return raw


def xavier_std(in_size, out_size):
    return np.sqrt(2. / (in_size + out_size))


# def weight(shape, dev):
#     initial = tf.truncated_normal(shape=shape, stddev=dev)
#     return tf.Variable(initial)
#
# def bias(shape):
#     initial = tf.constant(1e-4, shape=shape)
#     return tf.Variable(initial)


# def build(name):
#     s = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, 4])
#     with tf.variable_scope('{}eval'.format(name)):
#         w1 = weight(shape=[8,8,4,16], dev=xavier_std(8*8*4,8*8*16))
#         w2 = weight(shape=[4,4,16,32], dev=xavier_std(4*4*16,4*4*32))
#         w3 = weight(shape=[2592,256], dev=xavier_std(2596,256))
#         w4 = weight(shape=[256, 1], dev=xavier_std(256,1))
#         w5 = weight(shape=[256, NUM_OF_ACTION], dev=xavier_std(256,NUM_OF_ACTION))
#         b1 = bias([16])
#         b2 = bias([32])
#         b3 = bias([256])
#         b4 = bias([1])
#         b5 = bias([NUM_OF_ACTION])
#         conv1 = tf.nn.relu(tf.nn.conv2d(s, w1, [1,4,4,1], 'VALID') + b1)
#         conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w2, [1, 2, 2, 1], 'VALID') + b2)
#         fc = tf.nn.relu(tf.matmul(conv2, w3) + b3)
#         v = tf.nn.softmax(tf.matmul(fc, w4) + b4) #[None, 1]
#         p = tf.matmul(fc, w5) + b5 #[None, NUM_OF_ACTIONS]
#         return s, w1, w2, w3, w4, w5, b1, b2, b3, b4, b5, v, p


class A3CNet:
    def __init__(self, name, is_master, lr, master):
        self.lr = lr
        self.sess = tf.Session()
        self.action = NUM_OF_ACTION
        self.name = name
        self.is_master = is_master
        self.master = master
        self.s, self.w1, self.w2, self.w3, self.w4, self.w5, self.b1, self.b2, self.b3, self.b4, self.b5, self.v, self.p = self.build(
            self.name)
        if self.is_master:
            with tf.variable_scope('mastergradupdate'):
                # self.action_not_one_hot = tf.placeholder(dtype=tf.int32, shape = [None])
                self.action_one_hot = tf.placeholder(dtype=tf.float32, shape=[None, NUM_OF_ACTION])
                self.Re = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                ########################################################################################
                self.A = self.Re - self.v
                self.logprob = tf.log(tf.reduce_sum(self.p * self.action_one_hot, axis=1, keep_dims=True) + 1e-10)
                self.policy_loss = -tf.multiply(tf.stop_gradient(self.A), self.logprob)
                # self.value_loss = self.A
                self.entropy = tf.reduce_sum(tf.multiply(self.p, tf.log(self.p + 1e-10)), axis=1, keep_dims=True)
                self.loss = tf.reduce_mean(self.policy_loss + 0.5 * tf.square(self.A) + 0.01 * self.entropy)
                self.learn_op = tf.train.RMSPropOptimizer(self.lr, decay=0.99).minimize(self.loss)
                ########################################################################################
        self.sess.run(tf.global_variables_initializer())

    def weight(self, shape, dev):
        initial = tf.truncated_normal(shape=shape, stddev=dev)
        return tf.Variable(initial)

    def bias(self, shape):
        initial = tf.constant(1e-5, shape=shape)
        return tf.Variable(initial)

    def build(self, name):
        s = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, 4])
        with tf.variable_scope('{}eval'.format(name)):
            w1 = self.weight(shape=[8, 8, 4, 16], dev=xavier_std(8 * 8 * 4, 8 * 8 * 16))
            w2 = self.weight(shape=[4, 4, 16, 32], dev=xavier_std(4 * 4 * 16, 4 * 4 * 32))
            w3 = self.weight(shape=[2592, 256], dev=xavier_std(2596, 256))
            w4 = self.weight(shape=[256, 1], dev=xavier_std(256, 1))
            w5 = self.weight(shape=[256, NUM_OF_ACTION], dev=xavier_std(256, NUM_OF_ACTION))
            b1 = self.bias([16])
            b2 = self.bias([32])
            b3 = self.bias([256])
            b4 = self.bias([1])
            b5 = self.bias([NUM_OF_ACTION])
            conv1 = tf.nn.relu(tf.nn.conv2d(s, w1, [1, 4, 4, 1], 'VALID') + b1)
            conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w2, [1, 2, 2, 1], 'VALID') + b2)
            conv2_flat = tf.reshape(conv2, [-1, 2592])
            fc = tf.nn.relu(tf.matmul(conv2_flat, w3) + b3)
            v = tf.matmul(fc, w4) + b4  # [None, 1]
            p = tf.nn.softmax(tf.matmul(fc, w5) + b5)  # [None, NUM_OF_ACTIONS]
            # print('p: ', p.shape)
        return s, w1, w2, w3, w4, w5, b1, b2, b3, b4, b5, v, p

    def get_action(self):
        ob = self.state.copy()
        my_policy = self.sess.run(self.p, feed_dict={self.s: ob[np.newaxis, :]})
        # print(np.array(my_policy).reshape(4))
        act = np.random.choice(4, 1, p=np.array(my_policy).reshape(4))
        return act

    def set_init_state(self, ob):
        self.state = np.stack(ob, axis=2)

        # def build_net(self):
        #     conv1 = slim.conv2d(inputs=self.s, activation_fn=tf.nn.relu,
        #                         kernel_size=[8, 8], num_outputs=32, stride=[4, 4], padding='VALID')
        #     conv2 = slim.conv2d(inputs=conv1, activation_fn=tf.nn.relu,
        #                         kernel_size=[4, 4], num_outputs=64, stride=[2, 2], padding='VALID')
        #     conv3 = slim.conv2d(inputs=conv2, activation_fn=tf.nn.relu,
        #                         kernel_size=[3, 3], num_outputs=64, stride=[1, 1], padding='VALID')
        #     fc1 = slim.fully_connected(slim.flatten(conv3), num_outputs=512, activation_fn=tf.nn.relu)
        #     lstm1 = tf.contrib.rnn.BasicLSTMCell(num_units=256, state_is_tuple=True)
        #     ########################LSTM CELL DUMMY SETTING##########################
        #     # init_state = np.zeros((1, 256), np.float32)
        #     # init_state_tuple = tf.contrib.rnn.LSTMStateTuple(init_state, init_state)
        #     c_in = tf.placeholder(dtype=tf.float32, shape=[1, 256])
        #     h_in = tf.placeholder(dtype=tf.float32, shape=[1, 256])
        #     state_input = [c_in, h_in]
        #     lstm_output, lstm_state = tf.nn.dynamic_rnn(cell=lstm1,
        #                                                 inputs=fc1,
        #                                                 initial_state= state_input,
        #                                                 sequence_length=1,
        #                                                 time_major=False)
        #     p = slim.fully_connected(slim.flatten(lstm_output), num_outputs=NUM_OF_ACTION,
        #                                       activation_fn=tf.nn.softmax)
        #     v = slim.fully_connected(slim.flatten(lstm_output), num_outputs=1,
        #                                  activation_fn=None)


class Worker:
    def __init__(self, name, master):
        self.counter = 0
        self.master = master
        self.memory = deque()
        self.name = name
        # print('my name is!: {}, type: {}'.format(self.name, type(self.name)))

    def store_transition(self, memory_bar):
        self.memory.append(memory_bar)

    def work(self, tmax, gamma):
        global T
        # self.sess = tf.Session()
        self.env = gym.make('bo-v0').unwrapped
        self.net = A3CNet(name=self.name, is_master=False, lr=7e-4, master=self.master)
        # self.sess.run(tf.global_variables_initializer())
        for episode in range(TRAIN_EPISODE):
            self.memory = deque()
            reset_counter = 5
            total = 0
            steps = 0
            ob = self.env.reset()
            preprocessed_ob = preprocess(ob)
            ob_sequence = deque()
            ob_sequence.append(preprocessed_ob)
            ob_sequence.append(preprocessed_ob)
            ob_sequence.append(preprocessed_ob)
            ob_sequence.append(preprocessed_ob)
            self.net.set_init_state(ob_sequence)
            while True:
                store_flage = True
                # env.render()
                pre_ob = self.net.state
                action = self.net.get_action()
                lives = self.env.ale.lives()
                newob, reward, done, info = self.env.step(action)
                lost_one_live = (lives > self.env.ale.lives())
                total += reward
                preprocessed_newob = preprocess(newob)
                ob_sequence.append(preprocessed_newob)
                ob_sequence.popleft()
                self.net.set_init_state(ob_sequence)
                if steps < 4 or reset_counter < 4:
                    store_flage = False
                    reset_counter += 1
                if lost_one_live:
                    reset_counter = 0
                steps += 1
                if store_flage:
                    self.store_transition(
                        [pre_ob, np.stack(ob_sequence, axis=2), action, clipped_reward(reward), done or lost_one_live])
                terminated = done or lost_one_live
                if len(self.memory) == tmax or terminated:
                    # terminated = done or lost_one_live
                    # if len(self.memory) < tmax and terminated:
                    #     break
                    T += 1
                    pre_ob_batch = [d[0] for d in self.memory]
                    post_ob_batch = [d[1] for d in self.memory]
                    action_batch = [d[2] for d in self.memory]
                    reward_batch = [d[3] for d in self.memory]
                    done_batch = [d[4] for d in self.memory]
                    action_batch_one_hot = np.zeros((len(self.memory), NUM_OF_ACTION))
                    R_batch = deque()
                    if done or lost_one_live:
                        R = np.array([0.0])
                    else:
                        la = action_batch[-1]
                        R = self.net.sess.run(self.net.p,
                                              feed_dict={self.net.s: np.stack(ob_sequence, axis=2)[np.newaxis, :]})
                        R = R.reshape(-1)[la]
                        # print('Not done: ', R.shape)

                    n = len(self.memory)
                    for i in reversed(range(n)):
                        index = action_batch[i]
                        action_batch_one_hot[i][index] = 1.0
                        R = reward_batch[i] + gamma * R
                        R_batch.appendleft(R)

                    q.put([pre_ob_batch, post_ob_batch, action_batch_one_hot, R_batch])
                    self.memory = deque()
                    q.put(Syn(int(self.name)))  # Syn request
                    target_params = q_[int(self.name)].get()
                    self.net.sess.run([self.net.w1.assign(target_params[0]),
                                       self.net.w2.assign(target_params[1]),
                                       self.net.w3.assign(target_params[2]),
                                       self.net.w4.assign(target_params[3]),
                                       self.net.w5.assign(target_params[4]),
                                       self.net.b1.assign(target_params[5]),
                                       self.net.b2.assign(target_params[6]),
                                       self.net.b3.assign(target_params[7]),
                                       self.net.b4.assign(target_params[8]),
                                       self.net.b5.assign(target_params[9])])
                if done:
                    print('episode {} reward: {}, episode steps: {}, total steps: {}'.format(episode, total, steps, T))
                    break


class Master:
    def __init__(self):
        self.sess = tf.Session()
        self.net = A3CNet('master', True, 7e-4, master=None)

        self.sess.run(tf.global_variables_initializer())

    def process(self):
        msg = q.get()
        if isinstance(msg, Syn):
            id = msg.id
            target_params = self.sess.run([self.net.w1, self.net.w2, self.net.w3, self.net.w4, self.net.w5, self.net.b1,
                                           self.net.b2, self.net.b3, self.net.b4, self.net.b5])
            # print(type(target_params), type(target_params[0]))
            q_[id].put(target_params)
        else:
            pre_ob_batch = msg[0]
            post_ob_batch = msg[1]
            action_batch = msg[2]
            R_batch = msg[3]
            # print(np.array(R_batch).shape)
            self.sess.run(self.net.learn_op, feed_dict={self.net.action_one_hot: action_batch,
                                                        self.net.Re: R_batch,
                                                        self.net.s: np.array(pre_ob_batch)
                                                        })


def main():
    master = Master()
    workers = [Worker(str(i), master) for i in range(NUM_OF_WORKERS)]
    worker_threads = []
    for w in workers:
        # job = lambda: w.work(tmax=5, gamma=0.9)
        t = threading.Thread(target=w.work, args=(5, 0.9))
        t.start()
        worker_threads.append(t)
    while True:
        master.process()


if __name__ == '__main__':
    main()