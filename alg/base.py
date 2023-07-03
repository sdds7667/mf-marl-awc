import tensorflow as tf
import numpy as np
from wind_farm_gym import WindFarmEnv


class ValueNet:
    def __init__(self, sess, env: WindFarmEnv, handle, name, update_every=5, use_mf=False, learning_rate=1e-5, tau=0.05,
                 gamma=0.95, radius=(1, 1)):
        # assert isinstance(env, GridWorld
        self.env = env
        self.name = name
        self._saver = None
        self.sess = sess

        self.handle = handle
        self.view_space = (radius[0] * 2 + 1, radius[1] * 2 + 1, 3)
        self.action_view_space = (radius[0] * 2 + 1, radius[1] * 2 + 1)
        self.radius = radius
        self.feature_space = (3,)
        self.num_actions = 3
        self.act_n = env.n_turbines

        self.update_every = update_every
        self.use_mf = use_mf  # trigger of using mean field
        self.temperature = 0.1

        self.lr = learning_rate
        self.tau = tau
        self.gamma = gamma

        with tf.compat.v1.variable_scope(name or "ValueNet"):
            self.name_scope = tf.compat.v1.get_variable_scope().name
            self.obs_input = tf.compat.v1.placeholder(tf.float32, (None,) + self.view_space, name="Obs-Input")
            self.feat_input = tf.compat.v1.placeholder(tf.float32, (None,) + self.feature_space, name="Feat-Input")

            if self.use_mf:
                self.act_prob_input = tf.compat.v1.placeholder(tf.float32, (None, ) + self.action_view_space, name="Act-Prob-Input")

            self.act_input = tf.compat.v1.placeholder(tf.int32, (None,), name="Act")
            self.act_one_hot = tf.one_hot(self.act_input, depth=self.num_actions, on_value=1.0, off_value=0.0)

            with tf.compat.v1.variable_scope("Eval-Net"):
                self.eval_name = tf.compat.v1.get_variable_scope().name
                self.e_q = self._construct_net(active_func=tf.nn.relu)
                self.predict = tf.nn.softmax(self.e_q / self.temperature)
                self.e_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                               scope=self.eval_name)

            with tf.compat.v1.variable_scope("Target-Net"):
                self.target_name = tf.compat.v1.get_variable_scope().name
                self.t_q = self._construct_net(active_func=tf.compat.v1.nn.relu)
                self.t_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                               scope=self.target_name)
                print(self.t_variables)

            with tf.compat.v1.variable_scope("Update"):
                self.update_op = [tf.compat.v1.assign(self.t_variables[i],
                                                      self.tau * self.e_variables[i] + (1. - self.tau) *
                                                      self.t_variables[i])
                                  for i in range(len(self.t_variables))]
            with tf.compat.v1.variable_scope("Optimization"):
                self.target_q_input = tf.compat.v1.placeholder(tf.compat.v1.float32, (None,), name="Q-Input")
                self.e_q_max = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.act_one_hot, self.e_q), axis=1)
                self.loss = tf.compat.v1.reduce_sum(tf.compat.v1.square(self.target_q_input - self.e_q_max) / 3.0)
                self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _construct_net(self, active_func=None, reuse=False):
        flatten_obs = tf.compat.v1.reshape(self.obs_input, [-1, np.prod([v for v in self.obs_input.shape[1:]])])

        h_obs = tf.compat.v1.layers.dense(flatten_obs, units=256, activation=active_func,
                                          name="Dense-Obs")
        h_emb = tf.compat.v1.layers.dense(self.feat_input, units=32, activation=active_func,
                                          name="Dense-Emb", reuse=reuse)

        concat_layer = tf.compat.v1.concat([h_obs, h_emb], axis=1)

        if self.use_mf:
            flatten_action_probability = tf.compat.v1.reshape(self.act_prob_input, [-1, np.prod([v for v in self.act_prob_input.shape[1:]])])
            prob_emb = tf.compat.v1.layers.dense(flatten_action_probability, units=64, activation=active_func, name='Prob-Emb')
            h_act_prob = tf.compat.v1.layers.dense(prob_emb, units=32, activation=active_func, name="Dense-Act-Prob")
            concat_layer = tf.compat.v1.concat([concat_layer, h_act_prob], axis=1)

        dense2 = tf.compat.v1.layers.dense(concat_layer, units=128, activation=active_func, name="Dense2")
        out = tf.compat.v1.layers.dense(dense2, units=64, activation=active_func, name="Dense-Out")

        q = tf.compat.v1.layers.dense(out, units=self.num_actions, name="Q-Value")

        return q

    @property
    def vars(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)

    def calc_target_q(self, **kwargs):
        """Calculate the target Q-value
        kwargs: {'obs', 'feature', 'prob', 'dones', 'rewards'}
        """
        feed_dict = {
            self.obs_input: kwargs['obs'],
            self.feat_input: kwargs['feature']
        }

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            feed_dict[self.act_prob_input] = kwargs['prob']

        t_q, e_q = self.sess.run([self.t_q, self.e_q], feed_dict=feed_dict)
        act_idx = np.argmax(e_q, axis=1)
        q_values = t_q[np.arange(len(t_q)), act_idx]

        target_q_value = kwargs['rewards'] + q_values.reshape(-1) * self.gamma

        return target_q_value

    def update(self):
        """Q-learning update"""
        try:
            self.sess.run(self.update_op)
        except Exception as e:
            print("Graph update error: {}".format(e))

    def act(self, **kwargs):
        """Act
        kwargs: {'obs', 'feature', 'prob', 'eps'}
        """
        feed_dict = {
            self.obs_input: kwargs['state'][0],
            self.feat_input: kwargs['state'][1]
        }

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            assert len(kwargs['prob']) == len(kwargs['state'][0])
            feed_dict[self.act_prob_input] = kwargs['prob']

        actions = self.sess.run(self.predict, feed_dict=feed_dict)
        actions = np.argmax(actions, axis=1).astype(np.int32)
        return actions

    def train(self, **kwargs):
        """Train the model
        kwargs: {'state': [obs, feature], 'target_q', 'prob', 'acts'}
        """
        feed_dict = {
            self.obs_input: kwargs['state'][0],
            self.feat_input: kwargs['state'][1],
            self.target_q_input: kwargs['target_q'],
        }

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            feed_dict[self.act_prob_input] = kwargs['prob']

        feed_dict[self.act_input] = kwargs['acts']
        _, loss, e_q = self.sess.run([self.train_op, self.loss, self.e_q_max], feed_dict=feed_dict)
        return loss, {'Eval-Q': np.round(np.mean(e_q), 6), 'Target-Q': np.round(np.mean(kwargs['target_q']), 6)}
