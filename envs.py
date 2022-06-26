import gym
import numpy as np
from gym.spaces.box import Box
import cv2



def create_atari_env(env_id):
    env = gym.make(env_id)
    if len(env.observation_space.shape) > 1:
        env = AtariRescale42x42(env)
        env = NormalizedEnv(env)
    return env


def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(frame, (80, 80))
    frame = frame.mean(2) #Gray
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0) #Levels
    return frame


class AtariRescale42x42(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 80, 80])

    def _observation(self, observation_n):
        return _process_frame42(observation_n)


class NormalizedEnv(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation_n):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
                          observation_n.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
                         observation_n.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))
        ret = (observation_n - unbiased_mean) / (unbiased_std + 1e-8)
        return np.expand_dims(ret, axis=0)
