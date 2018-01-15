import Utils
import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf
import pickle
import cvxpy
import os
import cdd

#Define a new TF session
sess = U.single_threaded_session()
sess.__enter__()

#Experiment Directory
directory = "/home/vrubies/Research/baselines/baselines/trpo_mpi/experiments/"
environ = "PointMass-v0"

#Name of the pickle file
picklefile_names = os.listdir(path=directory)
picklefile_names = [directory+name for name in picklefile_names if environ in name]
picklefile2 = '/home/vrubies/Research/DLS/saved_net.pkl'

#Load pre-trained policy and get environment
policies,env = Utils.load_policy_and_env_from_pickle(sess,picklefile_names)

# PARAMETERS for learning the dynamics
architecture = {"hid_size":20,"num_hid_layers":2,"activation":tf.nn.relu}
optimizer = tf.train.MomentumOptimizer
loss_func = tf.losses.mean_squared_error
total_grad_steps=3000
traj_len=100
episodes=40
batch_size=10
l_rate=0.01
momentum=0.95

#Learn dynamics model of autonomous system
dynamics,list_W,list_b = Utils.learn_dynamics_model(sess,env,policies,architecture,optimizer,loss_func,total_grad_steps,
                                      traj_len,episodes,batch_size,l_rate,True,momentum)



M,b = Utils.get_region(list_W,list_b,np.array([[0.0],[0.0],[0.0],[0.0]]))
M = np.array(M)
b = -np.array(b)[None].T       
avore = Utils.convert_HtoV(M,b)    
