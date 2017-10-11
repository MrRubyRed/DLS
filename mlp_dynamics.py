from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

class MlpDynamics(object):

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
    
    def _init(self, ob_space, hid_size, num_hid_layers, activation=tf.nn.tanh):

        self.hid_size = hid_size;
        self.num_hid_layers = num_hid_layers;

        self.activation = str(activation);

        self.state_in = U.get_placeholder(name="state_in", dtype=tf.float32, shape=[None] + list(ob_space.shape))
        self.state_next = U.get_placeholder(name="state_next", dtype=tf.float32, shape=[None] + list(ob_space.shape))
        
        last_out = self.state_in
        for i in range(num_hid_layers):
            last_out = activation(U.dense(last_out, hid_size, "dynfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.prediction = U.dense(last_out, ob_space.shape[0], "dynfinal%i"%(i+1), weight_init=U.normc_initializer(1.0))

        self.n_step = U.function([self.state_in], [self.prediction])

    def step(self, ob):
        return self.n_step(ob[None])
    def get_architecture(self):
        return self.hid_size,self.num_hid_layers,self.activation



