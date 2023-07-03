import os
import tensorflow as tf
import numpy as np

from . import base
from . import tools


class MFQ(base.ValueNet):
    def __init__(self, sess, name, handle, env, sub_len, eps=1.0, update_every=5, memory_size=2 ** 20, batch_size=64,
                 radius=(1, 1)):
        super().__init__(sess, env, handle, name, use_mf=True, update_every=update_every, radius=radius)

        config = {
            'max_len': memory_size,
            'batch_size': batch_size,
            'obs_shape': self.view_space,
            'feat_shape': self.feature_space,
            'act_n': self.num_actions,
            'use_mean': True,
            'sub_len': sub_len,
            'former_actions': (radius[0] * 2 + 1, radius[1] * 2 + 1),
        }

        self.train_ct = 0
        self.replay_buffer = tools.MemoryGroup(**config)
        self.update_every = update_every

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        batch_name = self.replay_buffer.get_batch_num()

        for i in range(batch_name):
            obs, feat, acts, act_prob, obs_next, feat_next, act_prob_next, rewards = self.replay_buffer.sample()
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, prob=act_prob_next)
            loss, q = super().train(state=[obs, feat], target_q=target_q, prob=act_prob, acts=acts)

            self.update()

            if i % 50 == 0:
                print('[*] LOSS:', loss, '/ Q:', q)
        return float(loss), q

    def save(self, dir_path, step=0):
        model_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.compat.v1.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "mfq_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.compat.v1.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "mfq_{}".format(step))
        saver.restore(self.sess, file_path)

        print("[*] Loaded model from {}".format(file_path))
