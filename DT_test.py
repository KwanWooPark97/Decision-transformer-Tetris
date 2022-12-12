import numpy as np
import torch
from DT import DecisionTransformer
from cpprb import ReplayBuffer
import torch.nn.functional as F
from env import TetrisApp
from collections import deque
import pygame
import copy
import time

cols = 7
rows = 14
ret = [[0] * cols for _ in range(rows+1)]

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

def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[:, -1] = x[:, -1]
    #print(x)
    #print(disc_cumsum)
    for t in reversed(range(x.shape[1]-1)):
        disc_cumsum[:, t] = x[:, t] + gamma * disc_cumsum[:, t+1]

    #print(disc_cumsum)
    #exit(1)
    return disc_cumsum

device_name = 'cuda'
device = torch.device(device_name)
print("device set to: ", device)


class trainer():
    def __init__(self):
        self.max_eval_ep_len = 1000  # max len of one evaluation episode
        self.num_eval_ep = 10  # num of evaluation episodes per iteration

        self.batch_size = 256  # training batch size
        self.lr = 1e-5  # learning rate
        self.wt_decay = 1e-4  # weight decay
        self.warmup_steps = 1000  # warmup steps for lr scheduler

        # total updates = max_train_iters x num_updates_per_iter
        self.max_train_iters = 100
        self.num_updates_per_iter = 200
        self.state_dim = 105
        self.act_dim = 28
        self.context_len = 20  # K in decision transformer
        self.n_blocks = 5  # num of transformer blocks
        self.embed_dim = 255  # embedding (hidden) dim of transformer
        self.n_heads = 5  # num of transformer heads
        self.dropout_p = 0.1  # dropout probability
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            n_blocks=self.n_blocks,
            h_dim=self.embed_dim,
            context_len=self.context_len,
            n_heads=self.n_heads,
            drop_p=self.dropout_p,
        ).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wt_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps + 1) / self.warmup_steps, 1)
        )
        self.model.load_state_dict(torch.load('save_model.pt'))
        self.model.eval()

    def test(self):
        env = TetrisApp()
        global_step=0
        EPISODES = 10
        scores=[]
        episodes=[]
        clline=[]
        pygame.init()
        for e in range(EPISODES):
            rtg_target = 26
            done = False
            score = 0.0
            env.start_game()
            time_step = 1
            state = pre_processing(env.gameScreen)
            raw_state = np.reshape(state, [rows + 1, cols])
            state_deq = deque([np.zeros_like(raw_state) for _ in range(20)], maxlen=20)
            action_deq = deque([np.zeros([28, ]) for _ in range(20)], maxlen=20)
            reward_deq = deque([0 for _ in range(20)], maxlen=20)
            time_step_deq = deque([0 for _ in range(20)], maxlen=20)
            #reward_deq.append(rtg_target)

            while not done and time_step <= 1000:
                state_deq.append(raw_state)

                time.sleep(0.2)
                global_step += 1

                state_preds, action_preds, return_preds = self.model.forward(
                    timesteps=torch.Tensor(np.array(time_step_deq)).to(device).unsqueeze(dim=0).long(),
                    states=torch.Tensor(np.array(state_deq)).to(device).unsqueeze(dim=0),
                    actions=torch.Tensor(np.array(action_deq)).to(device).unsqueeze(dim=0),
                    returns_to_go=torch.Tensor(np.array(reward_deq)).to(device).unsqueeze(dim=0).unsqueeze(dim=-1)
                )

                action=torch.argmax(action_preds[0,-1])
                act = action_preds[0, -1].cpu().detach().numpy()
                action_deq.append(act)
                time_step_deq.append(time_step)

                reward, _ = env.step(action)

                # 게임이 끝났을 경우에 대해 보상 -1
                if env.gameover:
                    done = True
                    reward = -2.0
                else:
                    done = False
                #rtg_target -= reward
                reward_deq.append(reward)
                next_state = pre_processing(env.gameScreen)

                raw_state = np.reshape(next_state, [rows + 1, cols])
                time_step += 1
                score += reward

            # 보상 저장 및 학습 진행 관련 변수들 출력
            scores.append(score)
            episodes.append(e)
            clline.append(env.total_clline)
            print("episode:", e, "score: %.2f" % score, "total_clline:", env.total_clline, "global_step:",
                  global_step)
        print(min(clline), max(clline), sum(clline) / len(clline))


if __name__ == '__main__':
    DTT = trainer()
    DTT.test()