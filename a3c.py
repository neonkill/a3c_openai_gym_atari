from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp


import numpy as np
import random
import time

from torch.autograd import Variable

import time
from collections import deque

import os
import torch
import torch.multiprocessing as mp
from envs import create_atari_env
from model import A3C
import shared_optim
import wandb
import argparse

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='BreakoutDeterministic-v4', metavar='ENV',
                    help='environment to train on (default: BreakoutDeterministic-v4)')

args = parser.parse_args()

wandb.login()
wandb.init(project='A3C-openaiGYM-Atari',name='AssaultDeterministic-v4', config=args, sync_tensorboard=True, settings=wandb.Settings(start_method='thread', console="off"))


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, optimizer=None):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = A3C(env.observation_space.shape[0], env.action_space)

    optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    
    model.train()
    state = env.reset()
    state = torch.from_numpy(state)
    
    done = True
    episode_length = 0
    agent_reward = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            hx = Variable(torch.zeros(1, 256))
        else:
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []
        #Take Action
        for step in range(args.num_steps):
            value, logit, hx = model(
                (Variable(state.unsqueeze(0)), hx))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1,keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).data

            log_prob = log_prob.gather(1, Variable(action))
            #Compute States and Rewards
            state, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= args.max_episode_length
            #Clip Rewards from -1 to +1
            reward = max(min(reward, 1), -1)
            agent_reward += reward
            if done:
                episode_length = 0
                state = env.reset()
            #Append rewards, value functions, advantage and state
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            env.render()
            if done:
                agent_name = 'agent_' + str(rank)
                wandb.log({agent_name : agent_reward})
                agent_reward = 0
                break
        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), hx))
            R = value.data
        #Calculating Gradients
        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        GAE = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            #Discounted Sum of Future Rewards + reward for the given state
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            # Generalized Advantage Estimataion(GAE)
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            GAE = GAE * args.gamma * args.tau + delta_t
            policy_loss = policy_loss - \
                log_probs[i] * Variable(GAE) - 0.01 * entropies[i]
        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)
        optimizer.step()
    env.close()

def test(rank, args, shared_model):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = A3C(env.observation_space.shape[0], env.action_space)
    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    
    reward_sum = 0
    done = True

    start_time = time.time()
    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            hx = Variable(torch.zeros(1, 256), volatile=True)
        else:
            hx = Variable(hx.data, volatile=True)

        value, logit, hx = model(
            (Variable(state.unsqueeze(0), volatile=True), hx))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1)[1].data.numpy()
        # print("action: ",action)

        state, reward, done, _ = env.step(action)
        # print("reward: ",reward)
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action)
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))
            wandb.log({'test_model_reward' : reward_sum})
            #Save Shared Weights and Weights when certain scores are achieved
            if reward_sum >= 20 and args.env_name == "PongDeterministic-v4":
                print("Finished")
                torch.save(model.state_dict(), ('./A3C(Pong-1).pkl'))
                torch.save(shared_model.state_dict(), ('./A3C(Shared-Pong-1).pkl'))
                break
            elif reward_sum >= 300 and args.env_name == "BreakoutDeterministic-v4":
                print("Finished")
                torch.save(model.state_dict(),('./A3C(Breakout).pkl'))
                torch.save(shared_model.state_dict(),('./A3C(Shared-Breakout).pkl'))
            elif reward_sum >= 2000 and args.env_name == "QbertDeterministic-v4":
                print("Finished")
                torch.save(model.state_dict(),('./A3C(Qbert).pkl'))
                torch.save(shared_model.state_dict(),('./A3C(Shared-Qbert).pkl'))
            elif reward_sum >= 5000 and args.env_name == "AssaultDeterministic-v4":
                print("Finished")
                torch.save(model.state_dict(),('./A3C(Assault).pkl'))
                torch.save(shared_model.state_dict(),('./A3C(Assault).pkl'))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            #Rest
            time.sleep(60)
        state = torch.from_numpy(state)
        





if __name__ == "__main__":
    #Number of thread per cpu cores
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['DISPLAY'] = ':1'
    args = parser.parse_args()
    
    
    torch.manual_seed(args.seed)
    env = create_atari_env(args.env_name)
    shared_model = A3C(
        env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

   
    optimizer = shared_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()
    


    processes = []
    #Test Run
    p = mp.Process(target=test, args=(args.num_processes, args, shared_model))
    p.start()
    processes.append(p)
    #Run as many incarnation of the network for a given enviroment
    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
