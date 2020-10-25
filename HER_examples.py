#!/home/kevin/Development/stable-baselines/venv/bin/python3

import tensorflow as tf
import numpy as np

from stable_baselines import HER, SAC, DQN, TD3 #DDPG
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.common.bit_flipping_env import BitFlippingEnv
from gym.envs import robotics
import gym
import highway_env


def main():
    model_list = [SAC, TD3]#, DDPG]#, DQN]
    env_string_list = ['FetchReach-v1', 'parking-v0']#'FetchPush-v1']
    env_list = [gym.make(env_string) for env_string in env_string_list]
    #env_string_list += ['BitFlippingEnv'] * 2
    #env_list += [BitFlippingEnv(40, continuous=model, max_steps=40 ) for model in model_list]
    goal_selection_strategy = 'future'
    #print(env_list)
    #print(env_string_list)
    for model in model_list:
        for env_string, env in zip(env_string_list, env_list):
            model_str = str(model)[-6:-2].strip('.')

            if model == SAC:
                model = HER('MlpPolicy', env, model, n_sampled_goal=4,
                        goal_selection_strategy=goal_selection_strategy, buffer_size=1000000,
                        ent_coef='auto', batch_size=256, gamma=0.95, learning_rate=0.001, learning_starts=1000,
                        random_exploration=0.0, policy_kwargs=dict(layers=[256, 256, 256]),
                        tensorboard_log='./logs/' + env_string + '_' + model_str, verbose=1)
            elif model == TD3:
                model = HER('MlpPolicy', env, model, n_sampled_goal=4,
                        goal_selection_strategy=goal_selection_strategy, buffer_size=1000000,
                        batch_size=256, gamma=0.95, learning_rate=0.001, learning_starts=1000,
                        tensorboard_log='./logs/' + env_string + '_' + model_str, verbose=1)
            model.learn(20000)
            model_path = './models/' + env_string + '_' + model_str
            model.save(model_path)

            #model = HER.load(model_path, env=env)

#    obs = env.reset()
#    for _ in range(100):
#        action, _ = model.predict(obs)
#        obs, reward, done, _ = env.step(action)

#        print(reward)

#        if done:
#            obs = env.reset()


if __name__ == '__main__':
    main()
