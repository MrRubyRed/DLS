import baselines.common.tf_util as U
import pickle
import tensorflow as tf
import gym
import numpy as np
import cvxopt as cvx
import cvxpy
import cdd

cvx.solvers.options['show_progress'] = False

class Region(object):
    
    def __init__(self,C,b,interior_point,depth,max_depth,parent=None,dataW=None,datab=None):
        self.C = C
        self.b = b
        self.parent = parent
        self.depth = depth
        self.max_depth = max_depth #necessary max depth?
        self.interior_point = interior_point
        self.dataW = dataW
        self.datab = datab
        self.Jacobian = None
        
        self.H = None
        self.E = None
        self.F = None
        self.A_dyn = None
        self.b_dyn = None
        
        if depth == max_depth:
            self.M = None
            self.b_ = None
        else:
            self.M,self.b_ = self.get_Mandb_(self.depth)
        self.children = []

    def get_Mandb_(self,depth):
        if self.parent == None:
            return self.dataW[depth],self.datab[depth]
        else:
            return self.parent.get_Mandb_(depth)

    def find_family(self,M=None,b=None): #Root should feed M = I, b = np.array([0,0,0..])
        if(self.depth == self.max_depth):
            return
        if(self.depth == 0):
            D = np.diag(np.ones(self.M.shape[1]))
            M = np.diag(np.ones(self.M.shape[1]))
            b = np.zeros((self.M.shape[1],))
        else:
            tmp = ((np.matmul(M,self.interior_point) + b[None].T)>=0)
            D = np.diag(np.array([int(i) for i in tmp]))
        W_new = np.matmul(np.matmul(self.M,D),M)
        self.Jacobian = np.copy(W_new)
        B_new = (np.matmul(np.matmul(self.M,D),b[None].T).T)[0] + self.b_
        #B_newer = np.matmul(b[None],np.matmul(D,self.M.T))[0] + self.b_
        #Reorient hyperplanes
        W_new_,B_new_ = self.reorient_hyper_planes(W_new,B_new,self.interior_point,False)
        #Pre-Filtering Step to remove redundant hyperplanes outside super boundary interior
        try:
            W_,B_ = self.hyperP_Pre_filter(W_new_,B_new_)

            if(W_.size > 0):
                #Append super boundaries
                W_ = np.concatenate((self.C,W_),axis=0)
                B_ = np.concatenate((self.b,B_),axis=0) 
            else:
                W_ = np.copy(self.C)
                B_ = np.copy(self.b)
        except ValueError:
            print("Something went wrong")
        self.find_children(W_,B_)
        for child in self.children:
            if(self.depth == 0): print("MegaRegions found: " + str(self.children.index(child)))
            child.find_family(W_new,B_new)

    def find_children(self,W,B):
        #W_ = np.concatenate((self.C,W),axis=0)
        #B_ = np.concatenate((self.b,B),axis=0)
        C,b,inout,interior_point = self.hyperP_filter(W,B)
        if not self.point_check(interior_point,self.depth+1):
            if(len(b)<5): print("Problem occured. Trying to bound region by less than minimum required number of hyperplanes.")
            self.add_child(C,b,interior_point)
            print("New Child Added in LEVEL "+str(self.depth)+" with " + str(len(b)) + " boundaries! Num. of Children = " + str(len(self.children)))
        else:
            #print("Already visited!")
            return
        
        for k in range(len(self.C),len(inout)): #check this is correct (skipping super boundaries)
            if inout[k]:
                W_tmp = np.copy(W)
                B_tmp = np.copy(B)
                W_tmp[[k],:] = -W_tmp[[k],:]
                B_tmp[k] = -B_tmp[k]
                self.find_children(W_tmp,B_tmp)
        
    def add_child(self,C_child,b_child,interior_point):
        self.children.append(Region(C_child,b_child,interior_point,self.depth+1,self.max_depth,parent=self))

    def reorient_hyper_planes(self,W,B,point,Flag):
        if Flag:
            tmp = ((np.matmul(W,point) + B[None].T)>=0)
        else:
            tmp = ((np.matmul(W,point) + B[None].T)<0)
        diag = np.diag(np.array([int(i) for i in tmp])*2 - 1)
        W = np.matmul(diag,W)
        B = np.matmul(diag,B)
        return W,B

    def shift_regions(self,shift):
        self.b = self.b + np.matmul(self.C,shift).T[0]
        self.interior_point_shifted = self.interior_point - shift
        
        if(len(self.children) > 0):
            for c in self.children:
                c.shift_regions(shift)

    def find_maxr(self,max_r):
         if len(self.children)>0:
            for c in self.children:
                c.find_maxr(max_r)
         else:
             if max_r[0] < len(self.C):
                 max_r[0] = len(self.C)

    def find_HEFAB(self,env,dynamics,shift,constraints,T,max_r):
        if len(self.children)>0:
            for c in self.children:
                c.find_HEFAB(env,dynamics,shift,constraints,T,max_r)
        else:
            tmp = (self.b < 0); #find positive entries of bias
            diag = np.diag(np.array([int(i) for i in tmp])*2 - 1) #create a matrix to shift hyperplanes that are negative
            H_ = np.matmul(diag,self.C)
            b_ = np.matmul(diag,self.b[None].T)
            self.H = np.concatenate((H_,b_),axis=1) #   <--------- H matrix
            tmp = ((np.matmul(H_,self.interior_point_shifted) + b_)>=0)
            diag = np.diag(np.array([int(i) for i in tmp]))
            diag_ = np.diag(np.array([int(i) for i in tmp])*2 - 1)
            self.F = np.matmul(diag,self.H)         
            filler = np.concatenate((np.eye(self.F.shape[1]-1),np.zeros((self.F.shape[1]-1,1))),axis=1)
            self.F = np.concatenate((self.F,filler),axis=0) #   <--------- F matrix
            
            extrafill = max_r - self.H.shape[0]
            self.F = np.concatenate((self.F,np.zeros((extrafill,self.F.shape[1]))),axis=0)
            
            G = np.matmul(diag_,self.H) 
            flag = False
            if all(self.b < 0):
                flag = True
                G = np.zeros(G.shape)
            self.E = G
            
            A = dynamics.get_Jacobian(self.interior_point.T[0])[0][0].T
            b = dynamics.step(self.interior_point.T[0])[0].T - np.matmul(A,self.interior_point)
            self.A_dyn = (A - np.eye(len(A)))/env.env.dt
            self.b_dyn = b/env.env.dt + np.matmul(self.A_dyn,shift)
            
            big_A = np.concatenate((self.A_dyn,self.b_dyn),axis=1)
            big_A = np.concatenate((big_A,np.zeros((1,big_A.shape[1]))),axis=0)
            
            if not flag:
                #self.T = cvxpy.Variable(1,self.F.shape[0])
                self.u = cvxpy.Variable(1,self.E.shape[0])
                self.w = cvxpy.Variable(1,self.E.shape[0])
                constraints.append(T*self.F*big_A + self.u*self.E == 0)
                constraints.append(T*self.F - self.w*self.E == 0)
                constraints.append(self.u > 0)
                constraints.append(self.w > 0)
            else:
                big_A = np.concatenate((self.A_dyn,np.zeros(self.b_dyn.shape)),axis=1)
                big_A = np.concatenate((big_A,np.zeros((1,big_A.shape[1]))),axis=0)                
                #self.T = cvxpy.Variable(1,self.F.shape[0])
                self.u = cvxpy.Variable(1,self.E.shape[0])
                self.w = cvxpy.Variable(1,self.E.shape[0])
                constraints.append(T*self.F*big_A + self.u*self.E == 0)
                constraints.append(T*self.F - self.w*self.E == 0)
                constraints.append(self.u > 0)
                constraints.append(self.w > 0)
        
    def point_check(self,point,depth_chk): #checks if a point is in a known region, POINT MUST BE R^{Nx1}!!!!!
        try:
            if (self.depth == depth_chk) and ((np.matmul(self.C,point) + self.b[None].T) < 0).all(): #if we reach depth
                return True
            elif (self.depth < depth_chk) and ((np.matmul(self.C,point) + self.b[None].T) < 0).all() and (self.children != []):
                for child in self.children:
                    if child.point_check(point,depth_chk) == True:
                        return True
        except ValueError:
            print("Ooooops...")
        return False
    
    def return_regions(self,point,depth_chk): #checks if a point is in a known region
        try:
            if (self.depth == depth_chk) and ((np.matmul(self.C,point) + self.b[None].T) < 0).all(): #if we reach depth
                return (self.C,self.b,self.interior_point)
            elif (self.depth < depth_chk) and ((np.matmul(self.C,point) + self.b[None].T) < 0).all() and (self.children != []):
                for child in self.children:
                    tmp = child.return_regions(point,depth_chk)
                    if tmp:
                        return (self.C,self.b,self.interior_point),tmp
        except ValueError:
            print("Ooooops...")
        return False    

    def region_id(self,point,depth_chk): #checks if a point is in a known region
        try:
            if (self.depth == depth_chk) and ((np.matmul(self.C,point) + self.b[None].T) < 0).all(): #if we reach depth
                return "#"
            elif (self.depth < depth_chk) and ((np.matmul(self.C,point) + self.b[None].T) < 0).all() and (self.children != []):
                i = 0
                for child in self.children:
                    tmp = child.region_id(point,depth_chk)
                    if tmp:
                        return str(i)+"-"+tmp
                    i += 1
        except ValueError:
            print("Ooooops...")
        return False  

    def hyperP_Pre_filter(self,W,b):
        num_c = len(W)
        
        W = W.astype(np.double)
        b = b.astype(np.double)
        
        lil_W = []
        lil_b = []
        G = cvx.matrix(self.C)
        h = cvx.matrix(-self.b)
        for k in range(num_c): #skip first entries of W which are super-boundaries
            c = cvx.matrix(-W[k,:,None])
            sol = cvx.solvers.lp(c, G, h)
            sol_ = -sol["primal objective"]
            #print(str(np.abs(sol_ - b[k])))
            if(sol_ == None):
                print("Whoops")
            if((sol_ + b[k]) > -1e-5):
                lil_W.append(W[k,:])
                lil_b.append(b[k])
        return np.array(lil_W),np.array(lil_b)

    def hyperP_filter(self,W,b):
        num_c = len(W)
        
        W = W.astype(np.double)
        b = b.astype(np.double)
        
        lil_W = []
        lil_b = []
        inout = []
        interior_point = []
        log = []
        G = cvx.matrix(W)
        h = cvx.matrix(-b)
        for k in range(num_c): #skip first entries of W which are super-boundaries
            c = cvx.matrix(-W[k,:,None])
            sol = cvx.solvers.lp(c, G, h)
            sol_ = sol["primal objective"]
            log.append(sol_)
            #print(str(np.abs(sol_ - b[k])))
            if(sol_ == None):
                print("Whoops")
            if(np.abs(sol_ - b[k]) < 1e-5):
                interior_point.append(np.array(sol["x"]))
                inout.append(True)
                lil_W.append(W[k,:])
                lil_b.append(b[k])
            else:
                inout.append(False)
        if(len(lil_b)<5): 
            print("Problem occured. Trying to bound region by less than minimum required number of hyperplanes.")
            print(str(W))
            print(str(b))
        interior_point = np.mean(interior_point,axis=0)    
        return np.array(lil_W),np.array(lil_b),tuple(inout),interior_point

def convert_HtoV(A,b):
    M = np.concatenate((b,-A),axis=1)
    mat = cdd.Matrix(M,number_type='float')
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    ext = poly.get_generators()
    ext = np.array(ext)
    return ext

# The region_list list containing tuples of the form ((C,b,ipoint),[...])
#def get_all_regions(list_W,list_b,bounds,center): #assumes all W are in negative wrt origin position
#    C = []
#    b = []
#    for i in range(len(bounds)):
#        v = np.zeros((len(bounds),))
#        v[i] = 1.0
#        C.append(v)
#        C.append(-v)
#        low,high = bounds[i]
#        b.append(low)
#        b.append(high)
#    C = np.array(C)
#    b = np.array(b)
#        
#    RegionTree = Region(C,b,center,0,len(list_W))
    
            
    
def all_sub_regions_detected(list_W,list_b):
    pass
        
# AUXILIARY FUNCTIONS =========================================================

#def Find_Lyapunov_Function(self,Region,)

def point_induced_hyperP(list_W,list_b,point):
    D = [np.diag(np.ones(len(point)))]
    C = []
    b = []
    y = point
    for k in range(len(list_W)-1):
        y = np.matmul(D[-1],y)
        y = np.matmul(list_W[k],y) + list_b[k][None].T
        bool_y = (y >= 0)
        D_tmp = np.diag(np.array([int(tmp) for tmp in bool_y]))
        D.append(D_tmp)
    for k in range(len(list_W)):
        C.append(np.diag(np.ones(list_W[0].shape[1])))
        for i in range(k+1):
            tmp = np.matmul(list_W[i],D[i])
            C[-1] = np.matmul(tmp,C[-1])

    tmp = np.zeros((D[0].shape[0],1))
    for k in range(len(list_W)):
        tmp = list_b[k][None].T + np.matmul(np.matmul(list_W[k],D[k]),tmp) 
        b.append(tmp)                   
            
    return D,C,b
    
def get_region(list_W,list_b,point):
    _,C,b = point_induced_hyperP(list_W,list_b,point)
    big_C = np.concatenate(C,axis=0)
    big_b = np.concatenate(b,axis=0)
    tmp = (np.matmul(big_C,point) + big_b <= 0)
    diag = np.diag(np.array([int(i) for i in tmp])*2 - 1)
    
    big_C = np.matmul(diag,big_C)
    big_b = np.matmul(diag,big_b)  

    lil_C,lil_b,_,_ = hyperP_filter(big_C,big_b)
            
    return lil_C,lil_b

# Given a set of hyperplanes defined by (C,b), this function eliminates redundant
# constraints of the set {x | Cx + b < 0}
def hyperP_filter(C,b):
    num_c = len(C)

    lil_C = []
    lil_b = []
    inout = []
    interior_point = []
    G = cvx.matrix(C)
    h = cvx.matrix(-b)
    for k in range(num_c):
        c = cvx.matrix(-C[k,:,None])
        sol = cvx.solvers.lp(c, G, h)
        #print(sol["primal objective"] - big_b[k,0])
        if(np.abs(sol["primal objective"] - b[k]) < 1e-6):
            interior_point.append(np.array(sol["x"]))
            inout.append(True)
            lil_C.append(C[k,:])
            lil_b.append(b[k,0])
        else:
            inout.append(False)
    
    interior_point = np.mean(interior_point,axis=0)    
    return lil_C,lil_b,inout,interior_point

def random_sampling_regions(list_W,list_b,num_points=1,low=-3.0,high=3.0):
    C_list = []
    b_list = []
    for i in range(num_points):
        point = np.random.uniform(low,high,(4,1))

        flag = True
        if len(C_list) > 0:
            for C_,b_ in zip(C_list,b_list):
                chk = ((np.matmul(C_,point) + b_) < 0)
                if chk.all():
                    flag = False
                    break
        
        if flag:
            C,b = get_region(list_W,list_b,point)
            C = np.array(C)
            b = np.array(b)[None].T
            C_list.append(C)
            b_list.append(b)
            
    return C_list,b_list

def traj_sampling_regions(list_W,list_b,env,policy,episodes=1,traj_len=1):
    points,_,_ = collect_traj(env,policy,episodes,traj_len)
    C_list = []
    b_list = []
    for i in range(episodes*traj_len):
        point = points[i,:,None]

        flag = True
        if len(C_list) > 0:
            for C_,b_ in zip(C_list,b_list):
                chk = ((np.matmul(C_,point) + b_) < 0)
                if chk.all():
                    flag = False
                    break
        
        if flag:
            C,b = get_region(list_W,list_b,point)
            C = np.array(C)
            b = np.array(b)[None].T
            C_list.append(C)
            b_list.append(b)
            
    return C_list,b_list


    
def get_activation(activation):
    if(activation.find("relu") >= 0):
        return tf.nn.relu
    if(activation.find("tanh") >= 0):
        return tf.nn.tanh
    if(activation.find("sigmoid") >= 0):
        return tf.nn.sigmoid
    
def collect_traj(env,policies,episodes,traj_len):
    observations = []
    actions = []
    target = []
    obs = env.reset()
    for i in range(episodes):
        policy = policies[np.random.randint(len(policies))]
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
                                    picklefile_names):    #pickliefile name -- str
    
    from baselines.ppo1.mlp_policy import MlpPolicy 
    
    #Load information from policy contained in pickle file
    policies = []
    env_initialized = False    
    ind = 0;
    for picklefile in picklefile_names:
        all_info = pickle.load(open(picklefile,"rb"))    
        env_id = all_info["env_id"]
        hid_size, num_hid_layers, activation = all_info["architecture"]
        params = all_info["NNparams"]
        param_names = all_info["param_names"]
        
        #Initialize env
        if not env_initialized:
            env = gym.make(env_id)
            env_initialized = True
        
        #Create policy object
        policy = MlpPolicy(name="pi"+str(ind), ob_space=env.observation_space, ac_space=env.action_space,
                hid_size=hid_size, num_hid_layers=num_hid_layers, activation=get_activation(activation))
        
        policies.append(policy)
        #Initialize variables in "pi" scope        
        pol_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="pi"+str(ind))
        dict_params = {v.name:v for v in pol_params}
        pol_init = tf.variables_initializer(pol_params)
        sess.run(pol_init)
        pol_params = [pol_param for pol_param in pol_params if "pol" in pol_param.name]

        #Load weights and biases to NN
        for p in range(len(params)):
            sess.run(pol_params[p].assign(params[p]));
            #sess.run(dict_params[param_names[p]].assign(params[p]));

        ind += 1;

    return policies,env

# =============================================================================
    
#TODO: Load a pretraind dynamical model
def load_dynamics_model( picklefile,
                         sess,                  #tensorflow sess
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
    dynamics = learn_dynamics_model(sess,env,policy,architecture,optimizer,loss_func,0,
                                      traj_len,episodes,batch_size,l_rate,False,mom)
    
    weights = pickle.load(open(picklefile,"rb"))[0]
    coll = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dyn')
    for p in range(len(coll)):
        sess.run(coll[p].assign(weights[p]))
        
    return dynamics

# =============================================================================

#Learn a dynamics model
#TODO: Learn x_{t+1} = f(x_t,pi*(x_t)) -- or -- Delta_x = x_{t+1}-x_{t} = f(x_t,pi*(x_t)) ??
def learn_dynamics_model(sess,                  #tensorflow sess
                         env,                   #environment
                         policies,                #policy object
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
    data1,data2,target = collect_traj(env,policies,episodes,traj_len)
    v_da1,v_da2,v_targ = collect_traj(env,policies,episodes//2,traj_len)
    
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

    params = sess.run(dyn_params)
    list_W = []
    list_b = []
    for i in range(len(params)):
        if i % 2 == 0:
            list_W.append(params[i].T)
        else:
            list_b.append(params[i].T)
                
    return dynamics,list_W,list_b

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
            
        
    
    
        
