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
#dynamics = Utils.learn_dynamics_model(sess,env,policy,architecture,optimizer,loss_func,total_grad_steps,
#                                      traj_len,episodes,batch_size,l_rate,False,momentum)

#Get all weights and biases
#Wb = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dyn')
#list_W = []
#list_b = []
#for i in range(len(Wb)):
#    if i % 2 == 0:
#        list_W.append(sess.run(Wb[i]).T)
#    else:
#        list_b.append(sess.run(Wb[i]).T)

#TODO: Compare regions identified at random vs by following trajectories!
#stats,_,_ = Utils.collect_traj(env,policy,30,30)

#print("Start #1")
#C_list,b_list = Utils.random_sampling_regions(list_W,list_b,num_points=10000)
#print("Start #2")
#C_list_,b_list_ = Utils.traj_sampling_regions(list_W,list_b,env,policy,episodes=500,traj_len=20)

bounds = [(-10.0,10.0),(-10.0,10.0),(-10.0,10.0),(-10.0,10.0)]
C = []
b = []
for i in range(len(bounds)):
    v = np.zeros((len(bounds),))
    v[i] = 1.0
    C.append(v)
    C.append(-v)
    low,high = bounds[i]
    b.append(low)
    b.append(-high)
C = np.array(C)
b = np.array(b)

RegionTree = Utils.Region(C,b,np.zeros((4,1)),0,3)#len(list_W))
W = np.random.uniform(-3.0,3.0,(10,4))
b = np.random.uniform(-3.0,3.0,(10,))
tmp = (b <= 0)
diag = np.diag(np.array([int(i) for i in tmp])*2 - 1)
W = np.matmul(diag,W)
b = np.matmul(diag,b)
RegionTree.find_children(W,b)
#colors = []
#points = []
#for i in range(10000):
#
#    point = np.random.uniform(-2.0,2.0,(4,1))
#    #point[2:] = 0
#    j = 0
#    for C_,b_ in zip(C_list,b_list):
#        chk = ((np.matmul(C_,point) + b_) < 0)
#        if chk.all():
#            points.append(point.T)
#            colors.append(j)
#            break
#        j += 1

       
    
