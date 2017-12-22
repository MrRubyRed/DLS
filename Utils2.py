import baselines.common.tf_util as U
import pickle
import tensorflow as tf
import gym
import numpy as np

# AUXILIARY FUNCTIONS =========================================================

def get_activation(activation):
    if(activation.find("relu") >= 0):
        return tf.nn.relu
    if(activation.find("tanh") >= 0):
        return tf.nn.tanh
    if(activation.find("sigmoid") >= 0):
        return tf.nn.sigmoid
    
def collect_traj(env,policy,episodes,traj_len):
    observations = []
    actions = []
    target = []
    obs = env.reset()
    for i in range(episodes):
        for j in range(traj_len):
            action,predval = policy.act(False,obs)
            n_obs,rew,reset,info = env.step(action)
            observations.append(obs)
            actions.append(action)
            target.append(n_obs)
            if(reset):
                obs = env.reset()
            else:
                obs = n_obs
        obs = env.reset()
    
    return np.array(observations),np.array(actions),np.array(target)
        
                
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
                         l_rate=0.1,            #learning rate
                         withInput=False,       #learn dynamics with input
                         mom=0.95):             #momentum
    
    from mlp_dynamics import MlpDynamics
    
    #Architecture of dynamics NN
    hid_size = architecture["hid_size"]
    num_hid_layers = architecture["num_hid_layers"]
    activation = architecture["activation"]
    
    dynamics = MlpDynamics(name="dyn", ob_space=env.observation_space, ac_space=env.action_space, 
                           hid_size=hid_size,num_hid_layers=num_hid_layers, activation=activation,withInput=withInput)

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
    data1,data2,target = collect_traj(env,policy,episodes,traj_len)
    v_da1,v_da2,v_targ = collect_traj(env,policy,episodes//2,traj_len)
    
    #Learn the dynamical system
    for i in range(total_grad_steps):
        tmp = np.random.randint(len(data1), size=batch_size)
        if withInput:
            l,_ = sess.run([loss,grad_step],{dynamics.state_in:data1[tmp],dynamics.action_in:data2[tmp],dynamics.state_next:target[tmp]})
            if(np.mod(i,20) == 0):
                val_loss = sess.run(loss,{dynamics.state_in:v_da1,dynamics.action_in:v_da2,dynamics.state_next:v_targ})
                print("Loss = " + str(l) + " || Validation = " + str(val_loss))
        else:
            l,_ = sess.run([loss,grad_step],{dynamics.state_in:data1[tmp],dynamics.state_next:target[tmp]})
            if(np.mod(i,20) == 0):
                val_loss = sess.run(loss,{dynamics.state_in:v_da1,dynamics.state_next:v_targ})
                print("Loss = " + str(l) + " || Validation = " + str(val_loss))            
                
    return dynamics

# =============================================================================
            
#This function finds the points lying in the separatig hyperplanes of the piecewise
#linear system between two points a and b representing states.
def bisection_point_finder(dynamics,                #dynamics object
                           env,                     #gym envirnoment object
                           hyperP_list,             #list to be filled with hyperplane points
                           points=None,             #tuple of points
                           interval_L2=None,
                           eps=0.01,                #tolerance
                           rec_depth=0):            #recurrence depth
    a,b = points
    c = (b+a)/2.0
    
    J_a = dynamics.get_Jacobian(a)[0]
    J_b = dynamics.get_Jacobian(b)[0]
    J_c = dynamics.get_Jacobian(c)[0]
    
    condition1 = (J_a==J_c).all()
    condition2 = (J_c==J_b).all()
    
    if(interval_L2 / 2.0**rec_depth < eps):
        if condition1 != condition2:
            hyperP_list.append((c,J_a,J_b))
            return
        else:
            return
    
    if condition1 and condition2:
        bisection_point_finder(dynamics,env,hyperP_list,points=(a,c),
                                    interval_L2=interval_L2,eps=eps,rec_depth=rec_depth+1)                    
        bisection_point_finder(dynamics,env,hyperP_list,points=(c,b),
                                    interval_L2=interval_L2,eps=eps,rec_depth=rec_depth+1)
    elif condition1 and not condition2:
        return bisection_point_finder(dynamics,env,hyperP_list,points=(c,b),
                                            interval_L2=interval_L2,eps=eps,rec_depth=rec_depth+1)
    elif not condition1 and condition2:
        return bisection_point_finder(dynamics,env,hyperP_list,points=(a,c),
                                            interval_L2=interval_L2,eps=eps,rec_depth=rec_depth+1)
    elif not condition1 and not condition2:
        new_norm1 = np.linalg.norm(c-a)
        bisection_point_finder(dynamics,env,hyperP_list,points=(a,c),
                                    interval_L2=new_norm1,eps=eps,rec_depth=0)
        new_norm2 = np.linalg.norm(b-c)                                    
        bisection_point_finder(dynamics,env,hyperP_list,points=(c,b),
                                    interval_L2=new_norm2,eps=eps,rec_depth=0)

# Function to update points belonging to a hyperplane
def update_point_dict(dictionary,
                      hyperP_list):
    
    for p,Ja,Jb in hyperP_list:
        if (tuple(Ja.flatten()),tuple(Jb.flatten())) in dictionary:
            dictionary[(tuple(Ja.flatten()),tuple(Jb.flatten()))].append(p)
        elif (tuple(Jb.flatten()),tuple(Ja.flatten())) in dictionary:
            dictionary[(tuple(Jb.flatten()),tuple(Ja.flatten()))].append(p)
        else:
            dictionary[(tuple(Ja.flatten()),tuple(Jb.flatten()))] = [p]
            
    return dictionary
            
#Function that constructs all hyperplanes between two points a and b. 
def hyperplane_construction(dynamics,
                            env,
                            points=None,
                            extra_points=0,
                            eps=0.01,
                            per_param=0.05):
    #Minimum number of points to construct hyper plane equal to number of dims
    min_points = dynamics.ob_space
    a,b = points
    interval_L2 = np.linalg.norm(a-b)
    hyperP_list = []
    bisection_point_finder(dynamics,env,hyperP_list=hyperP_list,points=(a,b),interval_L2=interval_L2,eps=eps)
    check = [(p[1],p[2]) for p in hyperP_list]
    r = range(len(hyperP_list))
    point_dict = {indx:[p[0]] for indx,p in zip(r,hyperP_list)}
    i = 0
    while i < (min_points + extra_points - 1):
        hyperP_list = []
        perturb = np.random.randn(dynamics.ob_space)*per_param
        a_ = a + perturb
        b_ = b + perturb
        bisection_point_finder(dynamics,env,hyperP_list=hyperP_list,points=(a_,b_),interval_L2=interval_L2,eps=eps)
        new_check = [(p[1],p[2]) for p in hyperP_list]
        if(np.array_equal(np.array(new_check),np.array(check))):
            i += 1
            for indx,p in zip(r,hyperP_list): point_dict[indx].append(p[0])
    list_normals = []
    list_biases = []
    for i in range(len(point_dict)):
        A = np.array(point_dict[i])
        mean = np.mean(A,axis=0)
        A = A - mean
        _,_,D=np.linalg.svd(np.matmul(A.T,A))
        normal = D[-1,:]
        normal = normal/(np.linalg.norm(normal)+0.00001)
        list_normals.append(normal)
        list_biases.append(np.inner(normal,mean))
    return list_normals,list_biases
            
        
    
    
        