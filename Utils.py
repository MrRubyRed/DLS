import baselines.common.tf_util as U
import pickle
import tensorflow as tf
import gym
import numpy as np
import cvxopt as cvx

# AUXILIARY FUNCTIONS =========================================================

def point_induced_hyperP(list_W,list_b,point):
    D = [np.diag(np.ones(len(point)))]
    C = []
    b = []
    y = point
    for k in range(len(list_W)-1):
        y = np.matmul(D[-1],y)
        y = np.matmul(list_W[k],y) + list_b[k]
        bool_y = (y >= 0)
        D_tmp = np.diag(np.array([int(tmp) for tmp in bool_y]))
        D.append(D_tmp)
    for k in range(len(list_W)):
        C.append(np.diag(np.ones(list_W[0].shape[0])))
        for i in range(k+1):
            tmp = np.matmul(list_W[i],D[i])
            C[-1] = np.matmul(tmp,C[-1])

    tmp = np.zeros((D[0].shape[0],1))
    for k in range(len(list_W)):
        tmp = list_b[k] + np.matmul(np.matmul(list_W[k],D[k]),tmp) 
        b.append(tmp)                   
            
    return D,C,b
    
def get_region(list_W,list_b,point):
    _,C,b = point_induced_hyperP(list_W,list_b,point)
    big_C = np.concatenate(C,axis=0)
    big_b = np.concatenate(b,axis=0)
    tmp = (np.matmul(big_C,point) + big_b <= 0)
    diag = np.diag(np.array([int(i) for i in tmp])*2 - 1)
    
    big_C = np.matmul(diag,big_C).T
    big_b = np.matmul(diag,big_b).T
    num_c = len(big_C)    

    onoff = []
    G = cvx.matrix(big_C)
    h = cvx.matrix(-big_b)
    for k in range(num_c):
        c = cvx.matrix(-big_C[:,k])
        sol = cvx.solvers.lp(c, G, h)
        if(-sol["primal objective"] == 0):
            onoff.append(True)
        else:
            onoff.append(False)
        
    

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
            data.append(np.concatenate([obs,action]))
            target.append(n_obs)
            if(reset):
                obs = env.reset()
            else:
                obs = n_obs;
        obs = env.reset();
    
    return np.array(data),np.array(target)
        
                
# =============================================================================
# =============================================================================

    
#Function to load pretrained NNpolicy and environment
# This function takes in a picklefile of a dict with the following format:
# {"env_id":"env-name","architecture":[width,depth,activ_fcn],"NNparams":[W0,b0,W1...],"param_names":["w0","b0",...]}
def load_policy_and_env_from_pickle(sess,           #tensorflow session
                                    picklefile):    #pickliefile name -- str
    
    from baselines.ppo1.mlp_policy import MlpPolicy 
    
    #Load information from policy contained in pickle file    
    all_info = pickle.load(open(picklefile,"rb"))    
    env_id = all_info["env_id"]
    hid_size, num_hid_layers, activation = all_info["architecture"]
    params = all_info["NNparams"]
    param_names = all_info["param_names"]
    
    #Initialize env
    env = gym.make(env_id)
    
    #Create policy object
    policy = MlpPolicy(name="pi", ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=hid_size, num_hid_layers=num_hid_layers, activation=get_activation(activation))

    #Initialize variables in "pi" scope        
    pol_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pi')
    dict_params = {v.name:v for v in pol_params}
    pol_init = tf.variables_initializer(pol_params)
    sess.run(pol_init)

    #Load weights and biases to NN
    for p in range(len(params)):
        sess.run(dict_params[param_names[p]].assign(params[p]));

    return policy,env

# =============================================================================
    
#TODO: Load a pretraind dynamical model
def load_dynamics_model(picklefile):
    pass

# =============================================================================

#Learn a dynamics model
#TODO: Learn x_{t+1} = f(x_t,pi*(x_t)) -- or -- Delta_x = x_{t+1}-x_{t} = f(x_t,pi*(x_t)) ??
def learn_dynamics_model(sess,                  #tensorflow sess
                         env,                   #environment
                         policy,                #policy object
                         architecture,          #architecture dictionary
                         optimizer,             #tf optimizer
                         loss_func,             #tf loss function
                         total_grad_steps=1000, #gradient steps
                         traj_len=100,          #trajectory length
                         episodes=20,           #number of episodes
                         batch_size=10,         #batch size
                         l_rate=0.1,            #learning ratw
                         mom=0.95):             #momentum
    
    from mlp_dynamics import MlpDynamics
    
    #Architecture of dynamics NN
    hid_size = architecture["hid_size"]
    num_hid_layers = architecture["num_hid_layers"]
    activation = architecture["activation"]
    
    dynamics = MlpDynamics(name="dyn", ob_space=env.observation_space, ac_space=env.action_space, hid_size=hid_size,num_hid_layers=num_hid_layers, activation=activation)

    #Initialize variables in "dyn" scope
    dyn_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dyn')
    dict_dyn = {v.name:v for v in dyn_params}
    dyn_init = tf.variables_initializer(dyn_params)
    sess.run(dyn_init)
    
    #Initialize optimizer and loss
    with tf.variable_scope("loss_optimizer_scope"):
        loss = loss_func(dynamics.state_next,dynamics.prediction)
        try:
            grad_step = optimizer(l_rate).minimize(loss)
        except TypeError:
            grad_step = optimizer(l_rate,mom).minimize(loss)

    loss_optimizer_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='loss_optimizer_scope')
    loss_opt_init = tf.variables_initializer(loss_optimizer_scope)
    sess.run(loss_opt_init)
    
    #Collect data for training and testing
    data,target = collect_traj(env,policy,episodes,traj_len)
    v_da,v_targ = collect_traj(env,policy,episodes//2,traj_len)
    
    #Learn the dynamical system
    for i in range(total_grad_steps):
        tmp = np.random.randint(len(data), size=batch_size)
        l,_ = sess.run([loss,grad_step],{dynamics.state_in:data[tmp],dynamics.state_next:target[tmp]})
        if(np.mod(i,20) == 0):
            val_loss = sess.run(loss,{dynamics.state_in:v_da,dynamics.state_next:v_targ})
            print("Loss = " + str(l) + " || Validation = " + str(val_loss))
            
    return dynamics

# =============================================================================
            
#TODO: Check with crafted system
def bisection_hyperplane_finder(dynamics,				#dynamics object
					  			env,					#gym envirnoment object
					  			hyperP_list,
					  			points=None,			#tuple of points
					  			interval_L2=None,
					  			eps=0.01,				#tolerance
					  			rec_depth=0):
	a,b = points
	c = (b+a)/2.0
	
	if(interval_L2 / 2.0**rec_depth < eps):
		hyperP_list.append(c)
		return
	
	J_a = dynamics.get_Jacobian(a)
	J_b = dynamics.get_Jacobian(b)
	J_c = dynamics.get_Jacobian(c)
	
	condition1 = (J_a==J_c).all()
	condition2 = (J_c==J_b).all()
	
	if condition1 and condition2:
		return None					#point a and b are in the same linear region
	elif condition1 and not condition2:
		return bisection_hyperplane_finder(dynamics,env,hyperP_list,points=(c,b),
											interval_L2=interval_L2,eps=eps,rec_depth=rec_depth+1)
	elif not condition1 and condition2:
		return bisection_hyperplane_finder(dynamics,env,hyperP_list,points=(a,c),
											interval_L2=interval_L2,eps=eps,rec_depth=rec_depth+1)
	elif not condition1 and not condition2:
		new_norm1 = np.linalg.norm(c-a)
		bisection_hyperplane_finder(dynamics,env,hyperP_list,points=(a,c),
									interval_L2=new_norm1,eps=eps,rec_depth=0)
		new_norm2 = np.linalg.norm(b-c)									
		bisection_hyperplane_finder(dynamics,env,hyperP_list,points=(c,b),
									interval_L2=new_norm2,eps=eps,rec_depth=0)