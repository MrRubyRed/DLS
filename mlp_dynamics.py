from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np
import itertools

class MlpDynamics(object):

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
    
    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, activation=tf.nn.tanh): #Decouple Action and State

        self.hid_size = hid_size;
        self.num_hid_layers = num_hid_layers;

        self.activation = str(activation);

        self.state_in = U.get_placeholder(name="state_in", dtype=tf.float32, shape=[None] + list(ob_space.shape))
        self.action_in = U.get_placeholder(name="action_in", dtype=tf.float32, shape=[None] + list(ac_space.shape))
        self.merge_in = tf.concat([self.state_in,self.action_in],axis=1)
        self.state_next = U.get_placeholder(name="state_next", dtype=tf.float32, shape=[None] + list(ob_space.shape))
        
        last_out = self.merge_in
        for i in range(num_hid_layers):
            last_out = activation(U.dense(last_out, hid_size, "dynfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.prediction = U.dense(last_out, ob_space.shape[0], "dynfinal%i"%(i+1), weight_init=U.normc_initializer(1.0))

        #TODO: Get jacobian by slicing
        #self.Jac = tf.gradients(self.prediction,self.state_in,name="Jacobian")[0];
        self.Jac = tf.stack([tf.gradients(y, self.merge_in)[0] for y in tf.unstack(self.prediction, axis=1)],axis=2)
        
        self.n_step = U.function([self.state_in,self.action_in], [self.prediction])
        self.Jacobian = U.function([self.state_in,self.action_in], [self.Jac])

    def step(self, ob, act):
        return self.n_step(ob[None],act[None])
    def get_architecture(self):
        return self.hid_size,self.num_hid_layers,self.activation
    def get_Jacobian(self, state):
        if(state.ndim == 1):
            return self.Jacobian(state[None])
        elif(state.ndim == 2):
            return self.Jacobian(state)
        else:
            print("Wrong Input State. Should be (n,) or (m,n)")
            
class Test_PieceWise_Linear_Dynamics(object):
    
    def __init__(self,boundary_list,matrix_list):
        self.boundary_list = boundary_list
        self.matrix_list = matrix_list
        self.region_id = list(itertools.product([-1,1],repeat=len(self.boundary_list)))
        
    def step(self, ob):
        pass
    
    def get_Jacobian(self, state):
        region_id = []
        for normal,bias in self.boundary_list:
            if(np.inner(state,normal) - bias <= 0.0):
                region_id.append(-1)
            elif(np.inner(state,normal) - bias > 0.0):
                region_id.append(1)
        region_id = tuple(region_id)
        return self.get_MatfromID(region_id)
        
        
    def get_MatfromID(self,region_id):        
        indx = self.region_id.index(region_id)
        return self.matrix_list[indx]

