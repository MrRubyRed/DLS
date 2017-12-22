import Utils
import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf
import pickle
import cvxpy

#Define a new TF session
sess = U.single_threaded_session()
sess.__enter__()

#Name of the pickle file
picklefile = "/home/vrubies/Research/baselines/baselines/trpo_mpi/experiments/experiment_2017-10-27 16:05:44.152839.pkl"
picklefile2 = '/home/vrubies/Research/DLS/saved_net.pkl'

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
dynamics = Utils.load_dynamics_model(picklefile2,sess,env,policy,architecture,optimizer,loss_func,total_grad_steps,
                                      traj_len,episodes,batch_size,l_rate,False,momentum)

scaling = 1.0
#Get all weights and biases
Wb = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dyn')
list_W = []
list_b = []
for i in range(len(Wb)):
    if i % 2 == 0:
        list_W.append(sess.run(Wb[i]).T/scaling)
    else:
        list_b.append(sess.run(Wb[i]).T/scaling)

#TODO: Compare regions identified at random vs by following trajectories!
#stats,_,_ = Utils.collect_traj(env,policy,30,30)

#print("Start #1")
#C_list,b_list = Utils.random_sampling_regions(list_W,list_b,num_points=10000)
#print("Start #2")
#C_list_,b_list_ = Utils.traj_sampling_regions(list_W,list_b,env,policy,episodes=500,traj_len=20)

bounds = [(-3.0,3.0),(-3.0,3.0),(-3.0,3.0),(-3.0,3.0)]
C = []
b = []
for i in range(len(bounds)):
    v = np.zeros((len(bounds),))
    v[i] = 1.0
    C.append(v/scaling)
    C.append(-v/scaling)
    low,high = bounds[i]
    b.append(low/scaling)
    b.append(-high/scaling)
C = np.array(C)
b = np.array(b)
#import pickle
#C,b,W_y,B_y = pickle.load(open("Traitors.pkl","rb"))
RegionTree = Utils.Region(C,b,np.zeros((4,1)),0,3,dataW=list_W,datab=list_b)
#RegionTree.hyperP_Pre_filter(W_y,B_y)
#(self,C,b,interior_point,depth,max_depth,parent=None,dataW=None,datab=None):
#W = np.random.uniform(-3.0,3.0,(10,4))
#b = np.random.uniform(-3.0,3.0,(10,))
#tmp = (b <= 0)
#diag = np.diag(np.array([int(i) for i in tmp])*2 - 1)
#W = np.matmul(diag,W)
#b = np.matmul(diag,b)
#W_,B_ = RegionTree.reorient_hyper_planes(list_W[0],list_b[0],RegionTree.interior_point,False)
#RegionTree.find_children(W_,B_)
RegionTree = pickle.load(open("saved_Tree.pkl","rb"))[0] #.find_family()
tmp_ = dynamics.step(np.array([0.0,0.0,0.0,0.0]))
for k in range(10000):
    tmp_ = dynamics.step(tmp_[0][0])
print(tmp_)
#A_eq = dynamics.get_Jacobian(tmp_[0][0])[0][0].T
#b_eq = dynamics.step(tmp_[0][0])[0].T - np.matmul(A_eq,tmp_[0].T)
#shift = -np.matmul(np.linalg.inv((A_eq - np.eye(len(A_eq)))/env.env.dt),b_eq/env.env.dt)
shift = tmp_[0].T
avore = RegionTree.return_regions(tmp_[0].T,3); avore = avore[1]; avore = avore[1]; avore = avore[1];
RegionTree.shift_regions(shift)
#max_r = [0]
#RegionTree.find_maxr(max_r)
constraints = []
T = cvxpy.Variable(1,max_r[0]+4)
RegionTree.find_HEFAB(env,dynamics,shift,constraints,T)#,max_r[0])
obj = cvxpy.Minimize(0)
Prob = cvxpy.Problem(obj,constraints)
Prob.solve()

for i in range(100):
    gf = np.random.randint(0,len(RegionTree.children))
    dd = np.random.randint(0,len(RegionTree.children[gf].children))
    ch = np.random.randint(0,len(RegionTree.children[gf].children[dd].children))
    x = RegionTree.children[gf].children[dd].children[ch].interior_point_shifted
    x_ = np.concatenate((x,np.ones((1,1))),axis=0)
    tmp = np.matmul(np.array(RegionTree.children[gf].children[dd].children[ch].u.value),
                             RegionTree.children[gf].children[dd].children[ch].E)
    dec = np.matmul(tmp,x_)

    print(str(dec))
#    
t = np.array(T.value).T
x = np.random.uniform(-3,3,size=(4,1))
for i in range(100):
    rid = RegionTree.region_id(x,3).split("-")
    gf = int(rid[0]); dd = int(rid[1]); ch = int(rid[2]);
    F = RegionTree.children[gf].children[dd].children[ch].F
    x_ = np.concatenate((x,np.ones((1,1))),axis=0)
    val = np.matmul(np.matmul(F.T,t).T,x_)
    print(val)
    x = dynamics.step(x.T[0])[0].T
    
    
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

       
    
