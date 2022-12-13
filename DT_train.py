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

        self.batch_size = 64  # training batch size
        self.lr = 1e-4  # learning rate
        self.wt_decay = 1e-4  # weight decay
        self.warmup_steps = 1000  # warmup steps for lr scheduler

        # total updates = max_train_iters x num_updates_per_iter
        self.max_train_iters = 1000
        self.num_updates_per_iter = 100
        self.state_dim = 105
        self.act_dim = 28
        self.context_len = 20  # K in decision transformer
        self.n_blocks = 5  # num of transformer blocks
        self.embed_dim = 255  # embedding (hidden) dim of transformer
        self.n_heads = 5  # num of transformer heads
        self.dropout_p = 0.4  # dropout probability
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


        replay_buffer.load_transitions('./data/data_buffer.npz')
        count=0
        for i_train_iter in range(self.max_train_iters):

            self.log_action_losses = []
            self.model.train()

            for i in range(self.num_updates_per_iter):
                samples = replay_buffer.sample(self.batch_size)

                timesteps, states, actions, reward, traj_mask = samples["time_step"],samples["obs"],samples["act"],samples["rtg"],samples["traj_mask"]
                #returns_to_go=discount_cumsum(reward,1.0)
                print(states.shape)
                exit(1)
                timesteps = torch.from_numpy(timesteps).to(device).long()  # B x T
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
                action_preds = action_preds.view(-1, self.act_dim)[traj_mask.view(-1, ) > 0]
                action_target = action_target.view(-1, self.act_dim)[traj_mask.view(-1, ) > 0]
                action_target_acc=torch.argmax(action_target,dim=1)
                action_preds_acc=torch.argmax(action_preds,dim=1)
                #acc=torch.zeros_like(action_preds_acc)
                acc=torch.where(action_target_acc==action_preds_acc, 1,0)
                acc_train=torch.sum(acc)/action_preds_acc.shape[0]

                action_loss = F.cross_entropy(action_preds, action_target_acc)

                self.optimizer.zero_grad()
                action_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
                self.optimizer.step()
                self.scheduler.step()
                print(count,"번째 train loss : ",action_loss, "학습 정확도 :" ,acc_train)
                self.log_action_losses.append(action_loss.detach().cpu().item())
                #print(count)
                count+=1
                if count %1000==0:
                    torch.save(self.model.state_dict(), 'save_model.pt')



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
            raw_state = state
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
                rtg_target -= reward
                reward_deq.append(rtg_target)
                next_state = pre_processing(env.gameScreen)

                raw_state = next_state
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
    DTT.run()