#!/home/kevin/Development/stable-baselines/venv/bin/python3

import tensorflow as tf
import numpy as np

from stable_baselines import HER, SAC
from stable_baselines.her import GoalSelectionStrategy, HERGOalEnvWrapper
from stable_baselines.common.bit_flipping_env import BitFlippingEnv
from stable_baselines.common.callbacks import BaseCallback



class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log additional Tensor
        if not self.is_tb_set:
            with self.model.graph.as_default():
                tf.summary.scalar('value_target', tf.reduce_mean(self.model.value_target))
                self.model_summary = tf.summary.merge_all()
            self.is_tb_set = True
        # Log scalar value (here a random variable)
        value = np.random.random()
        summary = tf.Summary(value=[tf.Summary.Value(tag='random_value', simple_value=value)])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True



model_class = SAC

env = BitFlippingEnv(8, continuous=model_class, max_steps=8)

goal_selection_strategy = 'final'

model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy, verbose=1)

model.learn(1000, callback=TensorboardCallback())

model.save('./her_bit_env_sac')

model = HER.load('./her_bit_env_sac', env=env)

for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    print(reward)

    if done:
        obs = env.reset()



