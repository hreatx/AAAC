import numpy as np
import tensorflow as tf
import multiprocessing
import threading
import Queue
import random
import cv2
import gym
import os
import pandas as pd
from collections import deque
import shutil

# import matplotlib.pyplot as plt

gym.envs.register(id='bo-v0', entry_point='gym.envs.atari:AtariEnv',
                  kwargs={'game': 'breakout', 'obs_type': 'image', 'frameskip': 4, 'repeat_action_probability': 0.0},
                  max_episode_steps=100000,
                  nondeterministic=False, )
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

###############################constant###################################
NUM_OF_ACTION = 4
T = 0
TRAIN_EPISODE = 100000
NUM_OF_WORKERS = 8  # multiprocessing.cpu_count()
LITTLE_CONST = 1e-7
LOG_DIR = './log'
OUT_G = False
MAX_STEPS = 4000000


###############################constant###################################

def clipped_error(x):
    # Huber loss
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


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


class A3CNet:
    def __init__(self, name, is_master, master):
        self.action = NUM_OF_ACTION
        self.name = name
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, 4])
        self.policy_batch = deque()
        self.loss_batch = deque()
        self.a_batch = deque()
        self.entropy_batch = deque()
        with tf.variable_scope('{}eval'.format(name)):
            self.w1 = self.weight(shape=[8, 8, 4, 32], dev=xavier_std(8 * 8 * 4, 8 * 8 * 32))
            self.w2 = self.weight(shape=[4, 4, 32, 64], dev=xavier_std(4 * 4 * 32, 4 * 4 * 64))
            self.w3 = self.weight(shape=[3, 3, 64, 64], dev=xavier_std(3 * 3 * 64, 3 * 3 * 64))
            self.w4 = self.weight(shape=[3136, 512], dev=xavier_std(3136, 512))
            self.w5 = self.weight(shape=[512, 1], dev=xavier_std(512, 1))
            self.w6 = self.weight(shape=[512, NUM_OF_ACTION], dev=xavier_std(512, NUM_OF_ACTION))
            self.b1 = self.bias([32])
            self.b2 = self.bias([64])
            self.b3 = self.bias([64])
            self.b4 = self.bias([512])
            self.b5 = self.bias([1])
            self.b6 = self.bias([NUM_OF_ACTION])
            self.conv1 = tf.nn.relu(tf.nn.conv2d(self.s, self.w1, [1, 4, 4, 1], 'VALID') + self.b1)
            self.conv2 = tf.nn.relu(tf.nn.conv2d(self.conv1, self.w2, [1, 2, 2, 1], 'VALID') + self.b2)
            self.conv3 = tf.nn.relu(tf.nn.conv2d(self.conv2, self.w3, [1, 1, 1, 1], 'VALID') + self.b3)
            self.conv3_flat = tf.reshape(self.conv3, [-1, 3136])
            self.fc = tf.nn.relu(tf.matmul(self.conv3_flat, self.w4) + self.b4)
            self.v = tf.matmul(self.fc, self.w5) + self.b5  # [None, 1]
            self.p = tf.nn.softmax(tf.matmul(self.fc, self.w6) + self.b6)  # [None, NUM_OF_ACTIONS]
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{}eval'.format(name))
        if not is_master:
            with tf.variable_scope('{}grad'.format(self.name)):
                # self.action_not_one_hot = tf.placeholder(dtype=tf.int32, shape = [None])
                self.action_one_hot = tf.placeholder(dtype=tf.float32, shape=[None, NUM_OF_ACTION])
                self.Re = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                self.A = self.Re - self.v
                self.logprob = tf.log(tf.reduce_sum(self.p * self.action_one_hot, axis=1, keep_dims=True) + 1e-10)
                self.policy_loss = -tf.multiply(tf.stop_gradient(self.A), self.logprob)
                # self.value_loss = self.A
                self.entropy = tf.reduce_sum(tf.multiply(self.p, tf.log(self.p + 1e-10)), axis=1, keep_dims=True)
                self.loss = tf.reduce_mean(self.policy_loss + clipped_error(self.A) + 0.01 * self.entropy)

                self.gd = tf.gradients(self.loss, self.params)
                self.apply_gd = L_OP.apply_gradients(zip(self.gd, master.params))
                self.pull = [t.assign(e) for t, e in zip(self.params, master.params)]
            if self.name == '2':
                self.r_a = tf.reduce_mean(self.A)
                self.r_p = tf.reduce_mean(self.policy_loss)
                self.r_e = tf.reduce_mean(self.entropy)
                self.a_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                self.e_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                self.l_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                self.p_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                self.re_ph = tf.placeholder(dtype=tf.float32)
                self.a_t = tf.reduce_mean(self.a_ph)
                self.e_t = tf.reduce_mean(self.e_ph)
                self.l_t = tf.reduce_mean(self.l_ph)
                self.p_t = tf.reduce_mean(self.p_ph)
                self.re_t = self.re_ph * 1.0
                tf.summary.scalar('value_loss', self.a_t)
                tf.summary.scalar('policy_loss', self.p_t)
                tf.summary.scalar('entropy', self.e_t)
                tf.summary.scalar('total_loss', self.l_t)
                tf.summary.scalar('episode_reward', self.re_t)
                self.merged = tf.summary.merge_all()
                self.writer = tf.summary.FileWriter(LOG_DIR)

    def weight(self, shape, dev):
        initial = tf.truncated_normal(shape=shape, stddev=dev)
        return tf.Variable(initial)

    def bias(self, shape):
        initial = tf.constant(1e-5, shape=shape)
        return tf.Variable(initial)

    def get_action(self, s):
        ob = s
        my_policy = SESS.run(self.p, feed_dict={self.s: ob[np.newaxis, :]})
        # print(np.array(my_policy).reshape(4))
        act = np.random.choice(4, 1, p=np.array(my_policy).reshape(4))
        return act


class Worker:
    def __init__(self, name, master):
        self.ep_re = deque()
        self.counter = 0
        self.memory = deque()
        self.name = name
        self.env = gym.make('bo-v0').unwrapped
        self.net = A3CNet(name=self.name, is_master=False, master=master)
        # print('my name is!: {}, type: {}'.format(self.name, type(self.name)))

    def store_transition(self, memory_bar):
        self.memory.append(memory_bar)

    def work(self, tmax, gamma):
        global T
        for episode in range(TRAIN_EPISODE):
            if T >= MAX_STEPS:
                break
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

            while (not COORD.should_stop()) and T < MAX_STEPS:
                store_flage = True
                # env.render()
                pre_ob = np.stack(ob_sequence, axis=2)
                action = self.net.get_action(pre_ob)
                lives = self.env.ale.lives()
                newob, reward, done, info = self.env.step(action)
                lost_one_live = (lives > self.env.ale.lives())
                total += reward
                preprocessed_newob = preprocess(newob)
                ob_sequence.append(preprocessed_newob)
                ob_sequence.popleft()
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
                    T += 1
                    pre_ob_batch = [d[0] for d in self.memory]
                    post_ob_batch = [d[1] for d in self.memory]
                    action_batch = [d[2] for d in self.memory]
                    reward_batch = [d[3] for d in self.memory]
                    done_batch = [d[4] for d in self.memory]
                    action_batch_one_hot = np.zeros((len(self.memory), NUM_OF_ACTION))
                    R_batch = deque()
                    if done_batch[-1]:
                        R = np.array([0.0])
                    else:
                        p_a = post_ob_batch[-1]
                        R = SESS.run(self.net.v, feed_dict={self.net.s: p_a[np.newaxis, :]})
                        R = R.reshape(-1)
                        # print('Not done: ', R.shape)

                    n = len(self.memory)
                    for i in reversed(range(n)):
                        index = action_batch[i]
                        action_batch_one_hot[i][index] = 1.0
                        R = reward_batch[i] + gamma * R
                        R_batch.appendleft(R)
                    if self.name != '2':
                        SESS.run(self.net.apply_gd,
                                 feed_dict={self.net.s: pre_ob_batch,
                                            self.net.Re: R_batch,
                                            self.net.action_one_hot: action_batch_one_hot
                                            })
                    if self.name == '2':
                        aa, ap, ae, al, _ = SESS.run(
                            [self.net.r_a, self.net.r_p, self.net.r_e, self.net.loss, self.net.apply_gd],
                            feed_dict={self.net.s: pre_ob_batch,
                                       self.net.Re: R_batch,
                                       self.net.action_one_hot: action_batch_one_hot
                                       })
                        self.net.loss_batch.append(al)
                        self.net.a_batch.append(aa)
                        self.net.policy_batch.append(ap)
                        self.net.entropy_batch.append(ae)
                        # self.net.writer.add_summary(summ, T)
                    if T > 1000 and T % 800000 == 0:
                        saver.save(SESS, 'a3cworker{}in{}'.format(self.name, T), global_step=T)
                    SESS.run(self.net.pull)

                    self.memory = deque()

                if done:
                    print(
                    'worker {}: episode {} reward: {}, episode steps: {}, total steps: {}'.format(self.name, episode,
                                                                                                  total, steps, T))
                    self.ep_re.append(total)

                    if self.name == '2':
                        # print(total)
                        qq_1 = np.array(self.net.a_batch).reshape(len(self.net.a_batch), 1)
                        qq_2 = np.array(self.net.entropy_batch).reshape(len(self.net.entropy_batch), 1)
                        qq_3 = np.array(self.net.loss_batch).reshape(len(self.net.loss_batch), 1)
                        qq_4 = np.array(self.net.policy_batch).reshape(len(self.net.policy_batch), 1)

                        # print(qq.shape)
                        summ = SESS.run(self.net.merged, feed_dict={self.net.a_ph: qq_1,
                                                                    self.net.e_ph: qq_2,
                                                                    self.net.l_ph: qq_3,
                                                                    self.net.p_ph: qq_4,
                                                                    self.net.re_ph: total})  # self.net.re_ph: total
                        self.net.writer.add_summary(summ, episode)
                    self.net.policy_batch = deque()
                    self.net.loss_batch = deque()
                    self.net.a_batch = deque()
                    self.net.entropy_batch = deque()
                    self.ep_re = deque()

                    break


if __name__ == '__main__':
    SESS = tf.Session()
    with tf.device('/cpu:0'):
        # global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(7e-4, T, 100000, 0.99)
        L_OP = tf.train.RMSPropOptimizer(learning_rate, epsilon=1e-1)
        master = A3CNet('master', True, None)
        workers = [Worker(str(i), master) for i in range(NUM_OF_WORKERS)]
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())
    if OUT_G:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)
    saver = tf.train.Saver()
    worker_threads = []
    for w in workers:
        # job = lambda: w.work(tmax=5, gamma=0.9)
        t = threading.Thread(target=w.work, args=(5, 0.99))
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
    # di = {i: workers[i].ep_re for i in range(NUM_OF_WORKERS)}
    # data = pd.DataFrame(di)
    # data.to_csv('ep_reward.csv')


    # plt.plot(x, ep_reward)
    # plt.xlabel('episode')
    # plt.ylabel('episode reward')
    # plt.show()
