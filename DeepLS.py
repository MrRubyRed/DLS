import Utils
import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf

#Define a new TF session
sess = U.single_threaded_session()
sess.__enter__()

#Name of the pickle file
picklefile = "/home/vrubies/Research/baselines/baselines/trpo_mpi/experiments/experiment_2017-10-27 16:05:44.152839.pkl"

#Load pre-trained policy and get environment
policy,env = Utils.load_policy_and_env_from_pickle(sess,picklefile)

# PARAMETERS for learning the dynamics
architecture = {"hid_size":10,"num_hid_layers":2,"activation":tf.nn.relu}
optimizer = tf.train.MomentumOptimizer
loss_func = tf.losses.mean_squared_error
total_grad_steps=3000
traj_len=100
episodes=40
batch_size=10
l_rate=0.01
momentum=0.95

#Learn dynamics model of autonomous system
dynamics = Utils.learn_dynamics_model(sess,env,policy,architecture,optimizer,loss_func,total_grad_steps,
                                      traj_len,episodes,batch_size,l_rate,False,momentum)

#obs = env.reset()
#policy.act(True,obs)

#dynamics.get_Jacobian(obs);
#letussee = dynamics.get_Jacobian(np.array([0.0,0.0,0.0,0.0]),policy)
##import mlp_dynamics

##dim = 4
#dynamics2 = mlp_dynamics.Test_PieceWise_Linear_Dynamics_Diff(dim)

#a = np.random.uniform(-5,5,(dim,))
#b = np.random.uniform(-5,5,(dim,))
#normals,biases = Utils.hyperplane_construction(dynamics2,env,points=(a,b),extra_points=10)

##point_dict = {}
##for k in range(1000):
##    a = np.random.uniform(-5,5,(dim,))
##    b = np.random.uniform(-5,5,(dim,))
##    interval_L2 = np.linalg.norm(a-b)
##    hyperP_list = []
##    Utils.bisection_point_finder(dynamics,env,hyperP_list=hyperP_list,points=(a,b),interval_L2=interval_L2,eps=0.01)
##    point_dict = Utils.update_point_dict(point_dict,hyperP_list)

#normals,biases = Utils.hyperplane_construction(dynamics2,env,points=(a,b),extra_points=10)

#interval_L2 = np.linalg.norm(a-b)
#hyperP_list = []
#Utils.bisection_hyperplane_finder(dynamics2,env,hyperP_list=hyperP_list,points=(a,b),interval_L2=interval_L2,eps=0.01)
