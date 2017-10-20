import Utils
import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf

#Define a new TF session
sess = U.single_threaded_session()
sess.__enter__()

#Name of the pickle file
picklefile = "/home/vrubies/Research/baselines/baselines/trpo_mpi/experiments/experiment_2017-10-10 15:30:23.634862.pkl"

#Load pre-trained policy and get environment
policy,env = Utils.load_policy_and_env_from_pickle(sess,picklefile)

# PARAMETERS for learning the dynamics
architecture = {"hid_size":20,"num_hid_layers":2,"activation":tf.nn.relu}
optimizer = tf.train.MomentumOptimizer
loss_func = tf.losses.mean_squared_error
total_grad_steps=1000
traj_len=100
episodes=20
batch_size=10
l_rate=0.01
momentum=0.95

#Learn dynamics model of autonomous system
#dynamics = Utils.learn_dynamics_model(sess,env,policy,architecture,optimizer,loss_func,total_grad_steps,traj_len,episodes,batch_size,l_rate,momentum)

#obs = env.reset()
#policy.act(True,obs)

#dynamics.get_Jacobian(obs);
import mlp_dynamics
boundary_list = [(np.array([1.0,0.0]),0.0),(np.array([0.0,1.0]),0.0)]
matrix_list = [np.array([[2.0,-1.0],[-1.0,2.0]]),np.array([[2.0,1.0],[1.0,2.0]]),np.array([[2.0,1.0],[1.0,2.0]]),np.array([[2.0,-1.0],[-1.0,2.0]])]
dynamics2 = mlp_dynamics.Test_PieceWise_Linear_Dynamics(boundary_list,matrix_list)
a = np.array([1.0,0.2])
b = np.array([-0.2,-1.0])
interval_L2 = np.linalg.norm(a-b)
hyperP_list = []
Utils.bisection_hyperplane_finder(dynamics2,env,hyperP_list=hyperP_list,points=(a,b),interval_L2=interval_L2,eps=0.01)