#from baselines.common import explained_variance, zipsame, dataset
#from baselines import logger
#import baselines.common.tf_util as U
#import tensorflow as tf, numpy as np
#from baselines.common import colorize
#from mpi4py import MPI
#from collections import deque
#from baselines.common.mpi_adam import MpiAdam
#from baselines.common.cg import cg
#from contextlib import contextmanager
import baselines.common.tf_util as U
import pickle
import tensorflow as tf
import gym
import numpy as np

# AUXILIARY FUNCTIONS ===============

def get_activation(activation):
    if(activation.find("relu") >= 0):
        return tf.nn.relu
    if(activation.find("tanh") >= 0):
        return tf.nn.tanh
    if(activation.find("sigmoid") >= 0):
        return tf.nn.sigmoid
    
def collect_traj(env,policy,episodes,traj_len):
    data = [];
    target = [];
    obs = env.reset();
    for i in range(episodes):
        for j in range(traj_len):
            action,predval = policy.act(False,obs)
            n_obs,rew,reset,info = env.step(action)
            data.append(obs)
            target.append(n_obs)
            if(reset):
                obs = env.reset()
            else:
                obs = n_obs;
        obs = env.reset();
    
    return np.array(data),np.array(target)
        
                

#Function to load pretrained NNpolicy and environment
# This function assumes a picklefile of a dict with the following format:
# {"env_id":"env-name","architecture":[width,depth,activ_fcn],"NNparams":[W0,b0,W1...],"param_names":["w0","b0",...]}
def load_policy_and_env_from_pickle(sess,
                                    picklefile):
    
    from baselines.ppo1.mlp_policy import MlpPolicy
    
    all_info = pickle.load(open(picklefile,"rb"))
    
    env_id = all_info["env_id"]
    hid_size, num_hid_layers, activation = all_info["architecture"]
    params = all_info["NNparams"]
    param_names = all_info["param_names"]
    
    env = gym.make(env_id)

    policy = MlpPolicy(name="pi", ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=hid_size, num_hid_layers=num_hid_layers, activation=get_activation(activation))
        
    pol_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pi')
    dict_params = {v.name:v for v in pol_params}
    pol_init = tf.variables_initializer(pol_params)
    sess.run(pol_init)

    for p in range(len(params)):
        sess.run(dict_params[param_names[p]].assign(params[p]));

    return policy,env
    

def load_dynamics_model(picklefile):
    pass

#Learn a dynamics model
def learn_dynamics_model(sess,
                         env,
                         policy,
                         architecture,
                         optimizer,
                         loss_func,
                         total_grad_steps=1000,
                         traj_len=100,
                         episodes=20,
                         batch_size=10,
                         l_rate=0.1,
                         mom=0.95):
    
    from mlp_dynamics import MlpDynamics
    
    hid_size = architecture["hid_size"]
    num_hid_layers = architecture["num_hid_layers"]
    activation = architecture["activation"]
    
    dynamics = MlpDynamics(name="dyn", ob_space=env.observation_space, hid_size=hid_size,num_hid_layers=num_hid_layers, activation=activation)

    dyn_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dyn')
    dict_dyn = {v.name:v for v in dyn_params}
    dyn_init = tf.variables_initializer(dyn_params)
    sess.run(dyn_init)
    
    with tf.variable_scope("loss_optimizer_scope"):
        loss = loss_func(dynamics.state_next,dynamics.prediction)
        try:
            grad_step = optimizer(l_rate).minimize(loss)
        except TypeError:
            grad_step = optimizer(l_rate,mom).minimize(loss)

    loss_optimizer_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='loss_optimizer_scope')
    loss_opt_init = tf.variables_initializer(loss_optimizer_scope)
    sess.run(loss_opt_init)
    
    data,target = collect_traj(env,policy,episodes,traj_len)
    
    for i in range(total_grad_steps):
        tmp = np.random.randint(len(data), size=batch_size)
        l,_ = sess.run([loss,grad_step],{dynamics.state_in:data[tmp],dynamics.state_next:target[tmp]})
        if(np.mod(i,20) == 0):
            print("Loss = " + str(l))
            
    return dynamics
            
    

def find_hyper_planes(dyn_m,state_bounds=None,setofpoints=None):
    pass