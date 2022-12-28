import numpy as np
import torch
from DT import DecisionTransformer #DT 파일에 있는 class를 가져옵니다.
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

    kwargs = get_default_rb_dict(size=150000)

    return ReplayBuffer(**kwargs)

def discount_cumsum(x, gamma): #return_to_go를 계산해줍니다. 하지만 저는 reward를 사용했습니다.
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
        self.wt_decay = 1e-5  # weight decay
        self.warmup_steps = 100  # warmup steps for lr scheduler
        # total updates = max_train_iters x num_updates_per_iter
        self.max_train_iters = 6
        self.num_updates_per_iter = 3000
        self.state_dim = 105
        self.act_dim = 28

        ################################################################
        # 모델을 무작정 크게하면 오히려 성능이 좋아짐 GPT의 기본적인 학습을 잊지 말자
        # PC의 성능 한계로 n_blocks를 크게하면 메모리 부족으로 안돌아감
        # embed_dim은 n_heads로 나눠 떨어지게끔 설정해야함
        ################################################################
        self.context_len = 20  # K in decision transformer
        self.n_blocks = 10  # num of transformer blocks 5
        self.embed_dim = 800  # embedding (hidden) dim of transformer 255
        self.n_heads = 8  # num of transformer heads 5
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

    def run(self):
        replay_buffer = get_replay_buffer()
        replay_buffer.load_transitions('./data/data_buffer_10.npz')#학습에 사용할 데이터를 가져옵니다.
        count=0
        for i_train_iter in range(self.max_train_iters+1):
            self.log_action_losses = []
            self.model.train()
            if i_train_iter == 1:
                replay_buffer.clear()
                replay_buffer.load_transitions('./data/data_buffer_20.npz')  # 학습에 사용할 데이터를 교체합니다.
            elif i_train_iter == 2:
                replay_buffer.clear()
                replay_buffer.load_transitions('./data/data_buffer_50.npz')  # 학습에 사용할 데이터를 교체합니다.
            elif i_train_iter == 3:
                replay_buffer.clear()
                replay_buffer.load_transitions('./data/data_buffer_100.npz')  # 학습에 사용할 데이터를 교체합니다.
            elif i_train_iter == 4:
                replay_buffer.clear()
                replay_buffer.load_transitions('./data/data_buffer_1000.npz')  # 학습에 사용할 데이터를 교체합니다.
            elif i_train_iter == 5:
                replay_buffer.clear()
                replay_buffer.load_transitions('./data/data_buffer_1000_2.npz')  # 학습에 사용할 데이터를 교체합니다.
            elif i_train_iter == 6:
                replay_buffer.clear()
                replay_buffer.load_transitions('./data/data_buffer.npz')  # 학습에 사용할 데이터를 교체합니다.
            for i in range(self.num_updates_per_iter):
                samples = replay_buffer.sample(self.batch_size) #replay_buffer에서 batch_size 만큼 sample을 가져옵니다.
                timesteps, states, actions, reward, traj_mask = samples["time_step"],samples["obs"],samples["act"],samples["rtg"],samples["traj_mask"]
                timesteps = torch.from_numpy(timesteps).to(device).long()  # B x T numpy 데이터를 tensor 형태로 바꿉니다.
                states = torch.from_numpy(states).to(device)  # B x T x state_dim
                actions = torch.from_numpy(actions).to(device)  # B x T x act_dim
                returns_to_go = torch.from_numpy(reward).to(device).unsqueeze(dim=-1)  # B x T x 1
                traj_mask = torch.from_numpy(traj_mask).to(device)  # B x T
                action_target = torch.clone(actions).detach().to(device)

                state_preds, action_preds, return_preds = self.model.forward(
                    timesteps=timesteps,
                    states=states,
                    actions=actions,
                    returns_to_go=returns_to_go
                )

                # only consider non padded elements
                action_preds = action_preds.view(-1, self.act_dim)[traj_mask.view(-1, ) > 0] #모델의 출력을 mask가 1인 부분만 뽑아옵니다.
                action_target = action_target.view(-1, self.act_dim)[traj_mask.view(-1, ) > 0] #데이터에서 action을 mask가 1인 부분만 뽑아옵니다.
                action_target_acc=torch.argmax(action_target,dim=1)
                action_preds_acc=torch.argmax(action_preds,dim=1)

                #acc=torch.zeros_like(action_preds_acc)
                acc=torch.where(action_target_acc==action_preds_acc, 1,0)
                acc_train=torch.sum(acc)/action_preds_acc.shape[0] #정답과 예측의 정확도를 계산합니다.

                action_loss = F.cross_entropy(action_preds, action_target_acc) #예측값과 정답값의 cross_entropy loss를 계산합니다.

                self.optimizer.zero_grad()
                action_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.3) #한번에 크게 변하는 것을 방지해줍니다.
                self.optimizer.step()
                self.scheduler.step()
                print(count,"번째 train loss : ",action_loss, "학습 정확도 :" ,acc_train)
                self.log_action_losses.append(action_loss.detach().cpu().item())
                #print(count)
                count+=1
                if count %1000==0:
                    torch.save(self.model.state_dict(), 'save_model.pt')

if __name__ == '__main__':
    DTT = trainer()
    DTT.run()