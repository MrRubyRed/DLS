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
total_grad_steps=10000
traj_len=75
episodes=100
batch_size=20
l_rate=0.01
momentum=0.95

#Learn dynamics model of autonomous system
dynamics,list_Wd,list_bd = Utils.learn_dynamics_model(sess,env,policies,architecture,optimizer,loss_func,total_grad_steps,
                                      traj_len,episodes,batch_size,l_rate,True,momentum)
dW = (list_Wd,list_bd)

policy = policies[-2]
params = sess.run([v for v in policy.get_trainable_variables() if "pol" in v.name])
list_Wp = []
list_bp = []
for i in range(len(params)):
    if i % 2 == 0:
        list_Wp.append(params[i].T)
    else:
        list_bp.append(params[i].T)
pW = (list_Wp,list_bp)

PT=Utils.Polytope_Tree(np.array([[-1.0,0.0]]).T,policy,dynamics,pW,dW,0.01)
#M,b = Utils.get_region(list_W,list_b,np.array([[4.0],[4.0],[0.0],[0.0]]))
#M = np.array(M)
#b = -np.array(b)[None].T       
#avore = Utils.convert_HtoV(M,b)
#polytope = Utils.HPolytope(M,b)    
