#from tetris_pg_new_train_PPO import TetrisApp
from env import TetrisApp
import copy
import numpy as np
import tensorflow as tf
import pygame
import random
import time
from cpprb import ReplayBuffer
from collections import deque


cols = 7
rows = 14
ret = [[0] * cols for _ in range(rows+1)]

def get_default_rb_dict(size):
    return {
        "size": size,
        "default_dtype": np.float32,
        "env_dict": {
            "obs": {                  #observation
                "shape": (20,15,7)},
            "act": {
                "shape": (20,28)},
            "rtg": {"shape": 20},
            "traj_mask": {"shape":20},
            "time_step":{"shape":20}}}

def get_replay_buffer():

    kwargs = get_default_rb_dict(size=100000)

    return ReplayBuffer(**kwargs)



def pre_processing(gameimage):
    #ret = np.uint8(resize(rgb2gray(gameimage), (40, 40), mode='constant')*255) # grayscale
    copy_image = copy.deepcopy(gameimage)
    ret = [[0] * cols for _ in range(rows+1)]
    for i in range(rows+1):
        for j in range(cols):
            if copy_image[i][j] > 0:
                ret[i][j] = 1
            else:
                ret[i][j] = 0

    ret = sum(ret, [])
    return ret

class PERNETWORK(tf.keras.Model):
    def __init__(self,state_size):
        super(PERNETWORK, self).__init__()
        self.state_size=state_size
        self.layer1 = tf.keras.layers.Conv2D(32, (5, 5), strides=(1, 1), activation='relu', padding='same',
                                       kernel_initializer='he_uniform',input_shape=self.state_size)
        self.layer2 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same',
                                       kernel_initializer='he_uniform')
        self.layer3 = tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation='relu', padding='same',
                                       kernel_initializer='he_uniform')
        self.layer4 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same',
                                       kernel_initializer='he_uniform')
        self.layer5 = tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation='relu', padding='same',
                                       kernel_initializer='he_uniform')
        self.pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid', data_format=None)

        self.layer_2_1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same',
                                         kernel_initializer='he_uniform')  ##
        self.layer_2_2 = tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation='relu', padding='same',
                                         kernel_initializer='he_uniform')  ##
        self.layer_2_3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same',
                                         kernel_initializer='he_uniform')
        self.pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format=None)

        self.layer_r = tf.keras.layers.Conv2D(32, (rows + 1, 1), strides=(1, 1), activation='relu', padding='same',
                                         kernel_initializer='he_uniform')
        self.layer_c = tf.keras.layers.Conv2D(32, (1, cols), strides=(1, 1), activation='relu', padding='same',
                                         kernel_initializer='he_uniform')

        self.pool_1_r = tf.keras.layers.Conv2D(32, (13, 1), strides=(1, 1), activation='relu', padding='same',
                                          kernel_initializer='he_uniform')
        self.pool_1_c = tf.keras.layers.Conv2D(32, (1, 5), strides=(1, 1), activation='relu', padding='same',
                                          kernel_initializer='he_uniform')

        self.pool_2_r = tf.keras.layers.Conv2D(32, (12, 1), strides=(1, 1), activation='relu', padding='same',
                                          kernel_initializer='he_uniform')
        self.pool_2_c = tf.keras.layers.Conv2D(32, (1, 4), strides=(1, 1), activation='relu', padding='same',
                                          kernel_initializer='he_uniform')

        self.merge_layer1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform')
        self.v1= tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform')
        self.a1=tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform')
        self.v2=tf.keras.layers.Dense(1, activation='linear', kernel_initializer='he_uniform')
        self.a2=tf.keras.layers.Dense(28, activation='softmax', kernel_initializer='he_uniform')
        self.flatten_layer =tf.keras.layers.Flatten()


    def call(self, state):
        #state = tf.keras.layers.Input(shape=(self.state_size[0], self.state_size[1], self.state_size[2],))

        x1 = self.layer1(tf.dtypes.cast(state, tf.float32))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.pool_1(x5)
        x7 = self.layer_2_1(x6)
        x8 = self.layer_2_2(x7)
        x9 = self.layer_2_3(x8)
        y1 = self.pool_2(x9)
        y2 = self.layer_r(tf.dtypes.cast(state, tf.float32))
        y3 = self.layer_c(tf.dtypes.cast(state, tf.float32))
        y4 = self.pool_1_r(x6)
        y5 = self.pool_1_c(x6)
        y6 = self.pool_2_r(x9)
        y7 = self.pool_2_c(x9)

        layer = self.flatten_layer(x5)
        layer_2 = self.flatten_layer(x9)
        pool_1 = self.flatten_layer(x6)
        pool_2 = self.flatten_layer(y1)
        layer_r = self.flatten_layer(y2)
        layer_c = self.flatten_layer(y3)
        pool_1_r = self.flatten_layer(y4)
        pool_1_c = self.flatten_layer(y5)
        pool_2_r = self.flatten_layer(y6)
        pool_2_c = self.flatten_layer(y7)

        merge_layer = tf.concat(
            [layer, layer_2, pool_1, pool_2, pool_1_c, pool_1_r, pool_2_c, pool_2_r, layer_c, layer_r], axis=1)
        merge_layer=self.merge_layer1(merge_layer)
        vlayer =self.v1(merge_layer)
        alayer =self.a1(merge_layer)
        vf = self.v2(vlayer)
        af = self.a2(alayer)
        return af,vf
class PPO(tf.keras.Model):
    def __init__(self, state_size,action_size):
        super(PPO,self).__init__()
        self.state_size = state_size
        self.action_size=action_size
        self.layer1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same',
                                             kernel_initializer='he_uniform', input_shape=self.state_size)
        self.layer2 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same',
                                             kernel_initializer='he_uniform')
        self.layer3 = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation='relu', padding='same',
                                             kernel_initializer='he_uniform')
        self.layer4= tf.keras.layers.Dense(512,activation='relu')
        self.layer44 = tf.keras.layers.Dense(256, activation='relu')



        self.layer5 = tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same',
                                             kernel_initializer='he_uniform', input_shape=self.state_size)
        self.layer6 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same',
                                             kernel_initializer='he_uniform')
        self.layer7 = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation='relu', padding='same',
                                             kernel_initializer='he_uniform')
        self.layer8 = tf.keras.layers.Dense(512, activation='relu')
        self.layer88 = tf.keras.layers.Dense(256, activation='relu')

        self.pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format=None)
        self.ouact = tf.keras.layers.Dense(self.action_size, activation="softmax")
        self.oucri = tf.keras.layers.Dense(1, activation="linear")
        self.flatten_layer = tf.keras.layers.Flatten()

    def call(self,state):
        state=tf.dtypes.cast(state, tf.float32)
        x1=self.layer1(state)
        x2=self.pool_1(x1)
        x3=self.layer2(x2)
        x4=self.pool_1(x3)
        x5=self.layer3(x4)
        x6=self.pool_1(x5)
        x6=self.flatten_layer(x6)
        x7=self.layer4(x6)
        x8=self.layer44(x7)

        a1 = self.layer5(state)
        a2 = self.pool_1(a1)
        a3 = self.layer6(a2)
        a4 = self.pool_1(a3)
        a5 = self.layer7(a4)
        a6 = self.pool_1(a5)
        a6 = self.flatten_layer(a6)
        a7 = self.layer8(a6)
        a8=self.layer88(a7)

        act=self.ouact(x8)
        cri=self.oucri(a8)


        return act,cri

class Actor(tf.keras.Model):
    def __init__(self, state_size,action_size):
        super(Actor,self).__init__()
        self.state_size = state_size
        self.action_size=action_size
        self.layer1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same',
                                             kernel_initializer='he_uniform', input_shape=self.state_size)
        self.layer2 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same',
                                             kernel_initializer='he_uniform')
        self.layer3 = tf.keras.layers.Conv2D(16, (1, 1), strides=(1, 1), activation='relu', padding='same',
                                             kernel_initializer='he_uniform')
        self.layer4= tf.keras.layers.Dense(64,activation='tanh')
        self.ou=tf.keras.layers.Dense(action_size,activation="softmax")
        self.pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format=None)
        self.flatten_layer = tf.keras.layers.Flatten()

    def call(self,state):
        state=tf.dtypes.cast(state, tf.float32)
        x1=self.layer1(state)
        x2=self.pool_1(x1)
        x3=self.layer2(x2)
        x4=self.pool_1(x3)
        x5=self.layer3(x4)
        x6=self.pool_1(x5)
        x6=self.flatten_layer(x6)
        x7=self.layer4(x6)
        out=self.ou(x7)

        return out


class DuelingDoubleDQNagent():
    def __init__(self):
        self.action_space = [i for i in range(7*4)]  # 28 grouped action : board 7x14
        self.action_size = len(self.action_space)
        self.state_size = (rows + 1, cols, 1)
        #self.model=tf.keras.models.load_model('C:/Users/CDSL_7/PycharmProjects/pythonProject/promotion_0104_(tetris)/new_DQN_tetris_model/saved_model.pb')
        #self.model = Actor(self.state_size,self.action_size)
        self.model=PPO(self.state_size,self.action_size)
        #self.model = tf.saved_model.load('./new_model/tf_save')
        self.model.load_weights('GOODMODEL')
        #self.model.load_weights('./success/dqn_150000_best/new_DQN_tetris_model_floor')
        #self.log_dir = 'C:/Users/user/PycharmProjects/pythonProject/promotion_0104_(tetris)/ten'
        #self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)


    def get_action(self, state):
        policy,_ = self.model(state)
        policy = np.array(policy)[0]
        #action = np.argmax(policy)
        action=np.argmax(policy)
        #action = np.random.choice(self.action_size, p=policy)
        return action, policy


class DQN():
    def __init__(self):
        self.agent = DuelingDoubleDQNagent()
        self.global_step = 0
        self.clline,self.scores, self.episodes = [], [],[]



    def run(self):

        replay_buffer = get_replay_buffer()



        env = TetrisApp()

        EPISODES = 3000

        pygame.init()

        key_actions = ["LEFT", "RIGHT", "UP", "DOWN"]

        for e in range(EPISODES):

            done = False
            score = 0.0
            env.start_game()
            time_step=1
            state = pre_processing(env.gameScreen)
            raw_state=tf.reshape(state, [rows + 1, cols])
            state = tf.reshape(state, [rows + 1, cols,1])
            state_deq = deque([np.zeros_like(raw_state) for _ in range(20)],maxlen=20)
            action_deq = deque([np.zeros([28,]) for _ in range(20)], maxlen=20)
            traj_mask_deq = deque([0 for _ in range(20)], maxlen=20)
            reward_deq = deque([0 for _ in range(20)], maxlen=20)
            time_step_deq = deque([0 for _ in range(20)], maxlen=20)
            rtg=1500
            while not done and time_step<=50:
                state_deq.append(raw_state)
                time.sleep(0.2)

                self.global_step += 1

                # action = self.agent.get_action(np.reshape(state, [1, rows + 1, cols, 1]))
                action ,policy=self.agent.get_action(tf.expand_dims(state,0))
                action_deq.append(policy)
                traj_mask_deq.append(1)
                time_step_deq.append(time_step)
                reward, _ = env.step(action)

                # 게임이 끝났을 경우에 대해 보상 -1
                if env.gameover:
                    done = True
                    reward = -2.0
                else:
                    done = False

                replay_buffer.add(obs=state_deq, act=action_deq, rtg=reward_deq, traj_mask=traj_mask_deq,time_step=time_step_deq)
                reward_deq.append(reward)
                next_state = pre_processing(env.gameScreen)
                raw_state=np.reshape(next_state, [rows + 1, cols])
                next_state = np.reshape(next_state, [rows + 1, cols,1])

                state = next_state
                time_step +=1
                score += reward

            # 보상 저장 및 학습 진행 관련 변수들 출력
            self.scores.append(score)
            self.episodes.append(e)
            self.clline.append(env.total_clline)
            print("episode:", e, "score: %.2f"  %score, "total_clline:", env.total_clline, "global_step:",
                  self.global_step)
        print(min(self.clline) , max(self.clline),sum(self.clline)/len(self.clline))
        replay_buffer.save_transitions("data_buffer")

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    DQN = DQN()
    DQN.run()