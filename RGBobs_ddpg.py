import gym
from gym import envs
import dVRL_simulator
from gym import Wrapper, spaces
import numpy as np
from skimage import transform
from skimage.color import rgb2gray


class RGBobs(Wrapper):
    def __init__(self, env):
        super(RGBobs, self).__init__(env)
        self.reset()
        dummy_obs = env.render('rgb')
        dummy_obs_gray = rgb2gray(dummy_obs)
        dummy_obs_resized = transform.resize(dummy_obs_gray, (64, 64))  # if I use 1 at the end, an error arises during training about trying to match (1,84,84,1) with (?,84,84,1)
        self.observation_space = spaces.Box(low = 0, high = 255, shape = (64, 64, 1), dtype=np.uint8)


    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self.env.render('rgb')
        obs_gray = rgb2gray(obs)
        obs = transform.resize(obs_gray, (64, 64))
        return obs[:, :, None]*255

    def step(self,action):
        obs, reward, done, info = self.env.step(action)
        obs = self.env.render('rgb')
        obs_gray = rgb2gray(obs)
        obs = transform.resize(obs_gray, (64, 64))
        return obs[:, :, None]*255, reward, done, info
