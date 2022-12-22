#from tetris_pg_new_train_PPO import TetrisApp
from env import TetrisApp  #테트리스 환경을 가져옵니다.
import copy
import numpy as np
import tensorflow as tf
import pygame #파이썬으로 게임 환경을 만들때 사용하는 라이브러리입니다.
import random
import time
from cpprb import ReplayBuffer #강화학습의 PER,HER,ReplayBuffer등을 구현해둔 라이브러리입니다.
from collections import deque #list 타입의 변수의 최대 길이를 정해주는 라이브러리입니다.

cols = 7
rows = 14
ret = [[0] * cols for _ in range(rows+1)]

def get_default_rb_dict(size): #replaybuffer에 들어갈 요소들과 크기를 정해줍니다.
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

    kwargs = get_default_rb_dict(size=150000) #replaybuffer를 만들어줍니다. 최대 크기는 size로 정해줍니다.

    return ReplayBuffer(**kwargs)

def pre_processing(gameimage): #테트리스 화면을 이진화 해줍니다.
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

class PPO(tf.keras.Model): #PPO 네트워크입니다. 복잡한 모델이지만 간단한 모델을 이용해서 학습해도 성능은 크게 차이 없습니다.
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

class Agent():
    def __init__(self):
        self.action_space = [i for i in range(7*4)]  # 28 grouped action : board 7x14
        self.action_size = len(self.action_space)
        self.state_size = (rows + 1, cols, 1)
        self.model=PPO(self.state_size,self.action_size)
        self.model.load_weights('GOODMODEL')  #학습된 모델을 불러옵니다.

    def get_action(self, state):
        policy,_ = self.model(state)  #모델의 출력으로 policy를 가져옵니다.
        policy = np.array(policy)[0]
        action=np.argmax(policy) #argmax로 가장 높은 확률을 가지는 action을 선택합니다.
        return action, policy

class Test_tetris():
    def __init__(self):
        self.agent = Agent()
        self.global_step = 0
        self.clline,self.scores, self.episodes = [], [],[]

    def run(self):

        replay_buffer = get_replay_buffer()
        env = TetrisApp()
        EPISODES = 30000
        pygame.init()
        for e in range(EPISODES):
            done = False
            score = 0.0
            env.start_game()
            time_step=1
            state = pre_processing(env.gameScreen)
            raw_state=tf.reshape(state, [rows + 1, cols])
            state = tf.reshape(state, [rows + 1, cols,1])
            state_deq = deque([np.zeros_like(raw_state) for _ in range(20)],maxlen=20) #deque 라이브러리를 이용해 최대 20개의 list 데이터를 저장합니다.
            action_deq = deque([np.zeros([28,]) for _ in range(20)], maxlen=20)
            traj_mask_deq = deque([0 for _ in range(20)], maxlen=20)
            reward_deq = deque([0 for _ in range(20)], maxlen=20)
            time_step_deq = deque([0 for _ in range(20)], maxlen=20)
            current=0
            while not done and replay_buffer.get_current_episode_len() <= 150000 and current<=15:
                state_deq.append(raw_state)
                time.sleep(0.2)
                self.global_step += 1
                action ,policy=self.agent.get_action(tf.expand_dims(state,0))
                action_deq.append(policy)
                traj_mask_deq.append(1)
                time_step_deq.append(time_step)
                reward, _ = env.step(action)

                if env.gameover:
                    done = True
                    reward = -2.0
                else:
                    done = False
                #replay buffer에 transformer 학습에 필요한 데이터들을 저장합니다.
                replay_buffer.add(obs=state_deq, act=action_deq, rtg=reward_deq, traj_mask=traj_mask_deq,time_step=time_step_deq)
                reward_deq.append(reward)
                next_state = pre_processing(env.gameScreen)
                raw_state=np.reshape(next_state, [rows + 1, cols])
                next_state = np.reshape(next_state, [rows + 1, cols,1])
                state = next_state
                time_step +=1
                score += reward
                current+=1

                if replay_buffer.get_current_episode_len() %10000==0:
                    replay_buffer.save_transitions("data_buffer") #replay buffer를 파일로 저장합니다.
            if replay_buffer.get_current_episode_len() >= 150000:
                replay_buffer.save_transitions("data_buffer")
                exit(1)
            # 보상 저장 및 학습 진행 관련 변수들 출력
            self.scores.append(score)
            self.episodes.append(e)
            self.clline.append(env.total_clline)
            print("episode:", e, "score: %.2f"  %score, "total_clline:", env.total_clline, "global_step:",
                  self.global_step)
        print(min(self.clline) , max(self.clline),sum(self.clline)/len(self.clline))

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
    test = Test_tetris()
    test.run()