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
    
    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, activation=tf.nn.tanh, withInput=False): #Decouple Action and State

        self.withInput = withInput

        self.hid_size = hid_size
        self.num_hid_layers = num_hid_layers

        self.activation = str(activation)
        self.ob_space = ob_space
        self.ac_space = ac_space

        if self.withInput:
            self.state_in = U.get_placeholder(name="state_in", dtype=tf.float32, shape=[None] + list(ob_space.shape))
            self.action_in = U.get_placeholder(name="action_in", dtype=tf.float32, shape=[None] + list(ac_space.shape))
            self.merge_in = tf.concat([self.state_in,self.action_in],axis=1)
            last_out = self.merge_in
        else:
            self.state_in = U.get_placeholder(name="state_in", dtype=tf.float32, shape=[None] + list(ob_space.shape))
            last_out = self.state_in
        
        self.state_next = U.get_placeholder(name="state_next", dtype=tf.float32, shape=[None] + list(ob_space.shape))
        
        self.list_Wb = []

        for i in range(num_hid_layers):
            tmp = U.dense(last_out, hid_size, "dynfc%i"%(i+1), weight_init=U.normc_initializer(1.0)) #VRR changed
            self.list_Wb.append(tmp)
            last_out = activation(tmp)
        self.prediction = U.dense(last_out, ob_space.shape[0], "dynfinal%i"%(i+1), weight_init=U.normc_initializer(1.0))
        
        #TODO: Get jacobian by slicing
        #self.Jac = tf.gradients(self.prediction,self.state_in,name="Jacobian")[0];
        if self.withInput:
            self.A = tf.stack([tf.gradients(y, self.state_in)[0] for y in tf.unstack(self.prediction, axis=1)],axis=2)
            self.B = tf.stack([tf.gradients(y, self.action_in)[0] for y in tf.unstack(self.prediction, axis=1)],axis=2)
            self.n_step = U.function([self.state_in,self.action_in], [self.prediction])
            self.Jacobian = U.function([self.state_in,self.action_in], [self.A,self.B])            
        else:
            self.A = tf.stack([tf.gradients(y, self.state_in)[0] for y in tf.unstack(self.prediction, axis=1)],axis=2)
            self.n_step = U.function([self.state_in], [self.prediction])
            self.Jacobian = U.function([self.state_in], [self.A])

    def step(self, ob, act=None):
        if self.withInput:
            return self.n_step(ob[None],act[None])
        else:
            return self.n_step(ob[None])
    def get_architecture(self):
        return self.hid_size,self.num_hid_layers,self.activation
    def get_Jacobian(self, state, action=None, policy=None):
        if self.withInput:
            #dUdx = policy.get_Jacobian(state[None])[0]
            #action,_ = policy.act(False, state)
            A,B = self.Jacobian(state[None],action[None])
            #return A + np.matmul(dUdx,B)
            return A,B
        else:
            A = self.Jacobian(state[None])
            return A
        
            
class Test_PieceWise_Linear_Dynamics(object):
    
    #This class creates a piecewise linear dynamical system with matrix A1 or A2
    # depending on whether the sum(sign(x) > 0) is even or odd 
    
    def __init__(self,dim):
        self.A_even = np.random.uniform(-1,1,(dim,dim))
        self.A_odd = np.random.uniform(-1,1,(dim,dim))
        self.ob_space = dim
        
    def step(self, ob):
        pass
    
    def get_Jacobian(self, state):
        oddoreven = np.sum(np.sign(state)>0)
        if(oddoreven % 2 == 0):
            return self.A_even
        else:
            return self.A_odd

class Test_PieceWise_Linear_Dynamics_Diff(object):
    
    #This class creates a piecewise linear dynamical system with 
    #regions that all have different matrices 
    
    def __init__(self,dim):
        self.ob_space = dim
        
    def step(self, ob):
        pass
    
    def get_Jacobian(self, state):
        mat = np.matmul(np.ones((len(state),1)),np.sign(state)[None])
        return mat
