import baselines.common.tf_util as U
import pickle
import tensorflow as tf
import gym
import numpy as np
import cvxopt as cvx
import cvxpy
import cdd
import matplotlib.pyplot as plt

cvx.solvers.options['show_progress'] = False

# ============== Class Definitions for
# 1.) Region
# 2.) HPolytope
# 3.) Polytope Tree

def simulate_dynamics(state,steps,policy,dynamics):
    #checks = []
    traj=[]
    new_state = np.copy(state)
    #traj.append(new_state)
    #node = self.root
    for i in range(steps):
        new_state = dynamics.step(new_state,policy.act(False,new_state)[0])[0][0]
        traj.append(new_state)
        #node = node.check_if_in_union_BFS(state)
        #if(node is None):
        #    checks.append(False)
        #else:
        #    checks.append(True)
    traj = np.array(traj)
    #plt.scatter(traj[:,0],traj[:,1],c='g')
    #plt.pause(0.01)
    return traj

class HPolytope(object):
    
    def __init__(self,state,policy,dynamics,pW,dW,dt):
        A,b,dyn_Af,dyn_c = self.get_polytope(state,policy,dynamics,pW,dW,dt)
        self._init(A,b,dyn_Af,dyn_c)
        
    def _init(self,A,b,dyn_A,dyn_b):
        self.A = A
        self.b = -b
        self.dyn_A = dyn_A 
        self.dyn_b = dyn_b  
        
        self.neighbors_found = False
        self.neighbors = []

        self.flag_visit = False; #Used for BFS
        
        VandR = self.convert_HtoV()
        bool_vec = (VandR[:,0] == 1)
        self.V = VandR[bool_vec,1:]
        self.R = VandR[~bool_vec,1:]
        
        #Get (non-trivial) points in each facet and exit facets
        self.points,self.exit_facets,tp = self.find_facet_points_and_exit_facets()
        self.points = np.concatenate(self.points)     
        self.check = (np.abs(np.diagonal(np.matmul(self.A,self.points.T) - self.b)) < 1e-6).all()
        if not self.check: print(np.abs(np.diagonal(np.matmul(self.A,self.points.T) - self.b)))
        #print(str(self.A))
        #print(str(self.b))
        print(str(self.check) + " _ " + str(tp) + "_" + str(np.linalg.eigvals(self.dyn_A)))
        #print(str(self.points))
        self.interior_point = np.mean(self.points,axis=0,keepdims=True)

    def convert_HtoV(self):
        M = np.concatenate((self.b,-self.A),axis=1)
        mat = cdd.Matrix(M,number_type='float')
        mat.rep_type = cdd.RepType.INEQUALITY
        poly = cdd.Polyhedron(mat)
        ext = poly.get_generators()
        ext = np.array(ext)
        return ext

    def find_facet_points_and_exit_facets(self): #(Non-Trivial Point)
        vertex_loc_mat = np.matmul(self.A,self.V.T) - self.b
        points = []
        exit_facets = []
        for i in range(len(vertex_loc_mat)):
            vertices_in_face_i = np.array([True if abs(val) < 1e-9 else False for val in vertex_loc_mat[i,:]])
            Verts = self.V[vertices_in_face_i].T
            mph = np.mean(self.V[vertices_in_face_i],axis=0,keepdims=True) #mid point hull
            rays_in_face_i = np.array([True if abs((np.matmul(self.A[i,None,:],(ray[None] + mph).T)[0] - self.b[i])[0]) < 1e-9 else False for ray in self.R])
            if(len(rays_in_face_i) > 0) and rays_in_face_i.any():
                for ray in self.R:
                    print(abs((np.matmul(self.A[i,None,:],(ray[None] + mph).T)[0] - self.b[i])[0]))
                print(rays_in_face_i)
                print('Yes')
                Rays = self.R[rays_in_face_i].T
                mrp = np.mean(self.R[rays_in_face_i],axis=0,keepdims=True)
            else:
                mrp = 0;
                
            points.append(mph+mrp)
            
            #Check implementation
            dyn_at_point = np.matmul(self.dyn_A,points[-1].T) + self.dyn_b
            #print(dyn_at_point)
            if(np.matmul(self.A[i,None,:],dyn_at_point) > 0):
                exit_facets.append(True);
            else:
                dyn_at_verts = np.matmul(self.dyn_A,Verts) + self.dyn_b
                all_vinner_prods = np.matmul(self.A[i,None,:],dyn_at_verts)
                if(len(rays_in_face_i) > 0) and rays_in_face_i.any():
                    dyn_at_rays = np.matmul(self.dyn_A,Rays)
                    all_rinner_prods = np.matmul(self.A[i,None,:],dyn_at_rays)
                else:
                    all_rinner_prods = np.array([-1.0])
                if((all_vinner_prods <= 0).all() and (all_rinner_prods <= 0).all()):
                    exit_facets.append(False)
                else:
                    exit_facets.append(True)
            
        return points,exit_facets,(0,0,0)           
                
    def get_polytope(self,state,policy,dynamics,pW,dW,dt):
        list_Wp,list_bp = pW                                    #Unpack weightd and biases for the policy
        Ap,bp = self.get_region(list_Wp,list_bp,state)          #Get H-representation Polytope for state
        K = policy.get_Jacobian(state.T)[0][0].T                #Get K (from u = Kx + d) [Takes in np.arrat.shape = (1,n)] [returns np.array.spahe = (k,n,n)]
        d = policy.act(False,state.T[0])[0][None].T - np.matmul(K,state) #Get d (from u = Kx + d) [Takes in np.arrat.shape = (n,)] [returns (np.array.shape=(n,),float)]
        
        action = np.matmul(K,state)+d                           #Get action
        state_action = np.concatenate((state,action),axis=0)    #Concatenate state and action
        list_Wd,list_bd = dW                                    #Unpack weights and biases for the dynamics
        Ad,bd = self.get_region(list_Wd,list_bd,state_action)   #Get H-representation Polytope for state-action
        
        dyn_A,dyn_B = dynamics.get_Jacobian(state.T[0],action.T[0])       #Get dynamic matrices
        dyn_A = dyn_A[0].T
        dyn_B = dyn_B[0].T
        c = dynamics.step(state.T[0],action.T[0])[0].T - np.matmul(dyn_A,state) - np.matmul(dyn_B,action) #Get dynamic bias
        
        # Generate (CONTINUOUS TIME) autonomous dynamics model
        dyn_Af = ((dyn_A - np.eye(len(dyn_A))) + np.matmul(dyn_B,K))/dt
        dyn_c = (np.matmul(dyn_B,d) + c)/dt
        
        #Get polytope where the autonomous dynamics are valid
        n = len(state)
        Ap_ = Ad[:,:n] + np.matmul(Ad[:,n:],K)
        bp_ = np.matmul(Ad[:,n:],d) + bd
        new_A = np.concatenate((Ap,Ap_),axis=0)
        new_b = np.concatenate((bp,bp_),axis=0) 
        A,b,_,_ = self.hyperP_filter(new_A,new_b)
        A = np.array(A)
        b = np.array(b)[None].T

        return A,b,dyn_Af,dyn_c

    def check_if_in_union_BFS(self,state):
        queue = []
        undo_queue = []
        queue.append(self) #chgd
        self.flag_visit = True #chgd
        
        while queue:
            
            poly = queue.pop(0)
            undo_queue.append(poly)
            
            if((np.matmul(poly.A,state[None].T) <= poly.b).all()):
                for p in undo_queue: p.flag_visit = False
                for p in queue: p.flag_visit = False
                return poly
            else:
                for neighbor in poly.neighbors: 
                    if neighbor.flag_visit is False:
                        neighbor.flag_visit = True
                        queue.append(neighbor)

        for p in undo_queue: p.flag_visit = False
        for p in queue: p.flag_visit = False
        return None          
        
    def point_induced_hyperP(self,list_W,list_b,point):
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
        
    def get_region(self,list_W,list_b,point): #Input: List of weights and biases in "common form" (i.e. b \in R^{nx1} NOT R^{1xn})
        _,C,b = self.point_induced_hyperP(list_W,list_b,point)
        big_C = np.concatenate(C,axis=0)
        big_b = np.concatenate(b,axis=0)
        tmp = (np.matmul(big_C,point) + big_b <= 0)
        diag = np.diag(np.array([int(i) for i in tmp])*2 - 1)
        
        big_C = np.matmul(diag,big_C)
        big_b = np.matmul(diag,big_b)  
    
        lil_C,lil_b,_,_ = self.hyperP_filter(big_C,big_b)
        lil_C = np.array(lil_C)
        lil_b = np.array(lil_b)
        lil_b = lil_b[:,None]
        
        return lil_C,lil_b
    
    # Given a set of hyperplanes defined by (C,b), this function eliminates redundant
    # constraints of the set {x | Cx + b < 0}
    def hyperP_filter(self,C,b):
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

    def find_facet_points_and_exit_facets_(self):
        exit_facets = []
        enter_facets = []
        mixed_facets = []
        points = []
        for i in range(len(self.A)):
            c_ = self.A[i,None,:]
            c_n = c_/(np.linalg.norm(c_) + 1e-17)
            A_ = np.concatenate((self.A,-self.A[i,None,:]),axis=0)
            b_ = np.concatenate((self.b,-self.b[i,None,:]),axis=0)
            G = cvx.matrix(A_)
            h = cvx.matrix(b_)
            
            c = cvx.matrix(np.matmul(c_,self.dyn_A)[0])
            _c = cvx.matrix(np.matmul(-c_,self.dyn_A)[0])
            sol = cvx.solvers.lp(c, G, h)
            sol_ = cvx.solvers.lp(_c,G,h)
            
            if sol["status"] == "optimal" and sol_["status"] == "optimal":
                val = sol["primal objective"]+np.matmul(c_,self.dyn_b)[0][0]
                val_ = sol_["primal objective"]+np.matmul(-c_,self.dyn_b)[0][0]
                x_s = np.array(sol["x"])
                
                if(val > 0):
                    exit_facets.append(True)
                    enter_facets.append(False)
                    mixed_facets.append(False)
                elif(val_ > 0):
                    exit_facets.append(False)
                    enter_facets.append(True)
                    mixed_facets.append(False) #It contains at least one state that enters
                else:
                    exit_facets.append(False)
                    enter_facets.append(False)
                    mixed_facets.append(True) #It contains at least one state that enters                    
            elif sol["status"] == "dual infeasible" and sol_["status"] == "optimal":
                print("UNBOUNDED POLYTOPE")
                val_ = sol_["primal objective"]+np.matmul(-c_,self.dyn_b)[0][0]
                x_s = np.array(sol_["x"])
                if(val_ > 0):
                    exit_facets.append(False)
                    enter_facets.append(True)
                    mixed_facets.append(False) #It contains at least one state that enters
                else:
                    exit_facets.append(False)
                    enter_facets.append(False)
                    mixed_facets.append(True) #It contains at least one state that enters
            elif sol_["status"] == "dual infeasible" and sol["status"] == "optimal":
                print("UNBOUNDED POLYTOPE")
                val = sol["primal objective"]+np.matmul(c_,self.dyn_b)[0][0]
                x_s = np.array(sol["x"])
                if(val > 0):
                    exit_facets.append(False)
                    enter_facets.append(True)
                    mixed_facets.append(False) #It contains at least one state that enters
                else:
                    exit_facets.append(False)
                    enter_facets.append(False)
                    mixed_facets.append(True) #It contains at least one state that enters                
            else:
                print('Unkown status')
                print(sol["status"])
                print(self.A)
                print(self.b)
                print(i)
                print(self.dyn_A)
                print(self.dyn_b)
                
            v = np.random.normal(size=(1,c_n.shape[1]))
            alpha = np.inner(c_n[0],v[0])
            v = v - alpha*c_n
            beta = np.divide(self.b - np.matmul(self.A,x_s),np.matmul(self.A,v.T)+1e-17)
            v_mat = np.matmul(v.T,beta.T) + x_s
            bool_mat = np.matmul(self.A,v_mat) - self.b < 1e-6
            inters = bool_mat.all(axis=0)
            inters[i] = False
            mean_beta = np.mean(beta[inters])
            #if(len(beta[inters]) != 2): print('Error: More than two intersections'+' ('+str(beta[inters])+')')
            
            points.append(x_s.T + mean_beta*v)
            
        possible_exits = [a or b for a,b in zip(exit_facets,mixed_facets)] 
        ef = np.sum(exit_facets)
        enf = np.sum(enter_facets)
        mf = np.sum(mixed_facets)           
        return points,possible_exits,(ef,enf,mf)
        
        
# AUXILIARY FUNCTIONS =========================================================

def convert_HtoV(A,b):
    M = np.concatenate((b,-A),axis=1)
    mat = cdd.Matrix(M,number_type='float')
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    ext = poly.get_generators()
    ext = np.array(ext)
    return ext

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
    
def get_region(list_W,list_b,point): #Input: List of weights and biases in "common form" (i.e. b \in R^{nx1} NOT R^{1xn})
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
    if policies != []:
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
    else:
        observations = []
        target = []
        obs = env.reset()
        for i in range(episodes):
            for j in range(traj_len):
                n_obs,rew,reset,info = env.step(np.zeros((env.unwrapped.Na,)))
                observations.append(obs)
                target.append(n_obs)
                if(reset):
                    obs = env.reset()
                else:
                    obs = n_obs
            obs = env.reset()

        return np.array(observations),np.array([0]),np.array(target)

        
                
# =============================================================================
# =============================================================================

def load_policy_and_env_from_pickle(sess,           #tensorflow session
                                    environ,
                                    picklefile_names):    #pickliefile name -- str

    from baselines.ppo1.mlp_policy import MlpPolicy

    #Load information from policy contained in pickle file
    policies = []
    env_initialized = False
    ind = 0;
    if picklefile_names != []:
        for picklefile in picklefile_names:
            all_info = pickle.load(open(picklefile,"rb"))
            env_id = environ
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
    else:
        #Initialize env
        if not env_initialized:
            env = gym.make(environ)
            env_initialized = True

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

#TODO: Account for the fact that getting deeper layer L-constants is not trivial
def get_active_inactive(c,r,list_W,list_b,list_W_,list_b_): 
    Lg = 1.0
    Ll = 1.0
    active_list = []
    for i in range(len(list_W)-1):
        W_ = list_W_[i]
        b_ = list_b_[i]
        W = list_W[i]
        b = list_b[i]
        tmp = (np.matmul(W,c) + b[None].T) > 0
        tmp2 = np.abs(np.matmul(W_,c) + b_[None].T) < r
        tmp3 = tmp | tmp2
        active_list.append(tmp3)
        
        dig_ = np.diag(np.array([int(j) for j in tmp3.T[0]]))
        _,V,_ = np.linalg.svd(W)
        _,V_,_ = np.linalg.svd(np.matmul(dig_,W))
        print("Gone from the max S. Value " + str(V[0]) + " to " + str(V_[0]) + "for layer " + str(i))
        Lg = Lg*V[0]
        Ll = Ll*V_[0]
        
        dig = np.diag(np.array([int(j) for j in tmp.T[0]]))
        c = np.matmul(dig,np.matmul(W,c) + b[None].T)
        r = V_[0]*r
    
    _,V,_ = np.linalg.svd(list_W[-1])    
    Ll = Ll*V[0]
        
    return active_list,Lg,Ll

def show_tube(dynamics,c,r,T,list_W,list_b,list_W_,list_b_):
    fig, ax = plt.subplots()
    ax.set_xlim((-5, 5))
    ax.set_ylim((-5, 5))
    
    for t in range(T):
        circle = plt.Circle((c[0,0],c[1,0]), r, color='b')
        ax.add_artist(circle)
        
        active_list,Lg,Ll = get_active_inactive(c,r,list_W,list_b,list_W_,list_b_)
        c = dynamics.step(c.T[0])[0].T
        r = r*Ll
        plt.pause(3)

def generate_constraints_and_vars(list_W,list_b,variables,constraints):
    log = []
    var_in = []
    var_out = variables
    for i in range(len(list_W)):
        W = list_W[i]
        b = list_b[i]
        N = W.shape[0]
        upper = np.zeros((N,))
        lower = np.zeros((N,))
        for j in range(len(W)):
            #print(W[j,:])
            objective_max = cvxpy.Maximize(W[j,:]*var_out[-1])
            objective_min = cvxpy.Minimize(W[j,:]*var_out[-1])
            
            prob_max = cvxpy.Problem(objective_max,constraints)
            prob_min = cvxpy.Problem(objective_min,constraints)
            
            tmp_max = prob_max.solve(solver=cvxpy.CVXOPT)
            #tmp_max = np.array(tmp.value)
            tmp_min = prob_min.solve(solver=cvxpy.CVXOPT)
            #tmp_min = np.array(tmp.value)
            
            upper[j] = tmp_max + b[j]
            lower[j] = tmp_min + b[j]
        
        log.append((lower,upper))
        lower  = np.array([min([0,tmp]) for tmp in lower])
        upper  = np.array([max([0,tmp]) for tmp in upper])
        print("Upper-Lower #"+str(i)+": ",end="")
        for u,l in zip(upper,lower):
            print(str(round(min(u,abs(l),u-l),2))+" ",end="")
        print("")
        if(min(np.abs(upper-lower)) == np.inf):
            print("Whoops..")
        A_ = np.diag(upper/(upper-lower))
        var_in.append(W*var_out[-1] + b)
        var_out.append(cvxpy.Variable(N,1))
        constraints.append(var_out[-1] >= 0)
        constraints.append(var_out[-1] >= var_in[-1])
        constraints.append(var_out[-1] <= A_*var_in[-1] - np.matmul(A_,lower)) #TODO check this constaint
    return var_in[-1]
            
        
def compute_supporting_planes(list_W,list_b,initial_set,num_planes=None, draw=False):
    N = list_W[0].shape[1]
    if num_planes == None:
        num_planes = N+1

    H = np.zeros((num_planes,N))
    B = np.zeros((num_planes,))

    constraints = []
    variables = []
    
    x = cvxpy.Variable(N,1)
    variables.append(x)
    
    A,b = initial_set
    constraints.append(A*x <= b)
    
    v = generate_constraints_and_vars(list_W,list_b,variables,constraints)

    ang = 2*np.pi/num_planes
    for i in range(num_planes):
        l = np.array([[np.cos(ang + ang*i),np.sin(ang + ang*i)]])#np.random.multivariate_normal(np.zeros((N,)),np.eye(N),size=(1,))
        #l = l/np.linalg.norm(l)
        objective = cvxpy.Maximize(l*v)

        prob = cvxpy.Problem(objective,constraints)
        result = prob.solve(solver=cvxpy.CVXOPT)#, verbose=True)
    
        h = l
        H[[i],:] = h
        B[i] = result
        
        if draw:
            graph_plane(h,result)
            
        x_support = np.array(variables[-1].value)
        #print(x_support)
    
    #Ab = np.concatenate((H,B),axis=1)
    plt.xlim((-10,10))
    plt.ylim((-10,10))    
    plt.pause(1.0)
    plt.clf()
    #plt.show()
    return H,B#convert_HtoV(H,B),convert_HtoV(H,-B)

def graph_plane(h,b):
    x = np.arange(-10.0, 10.0, 0.5)
    plt.plot(x, (b - h[0,0]*x)/h[0,1], color='blue')
    #plt.show()

def convert_HtoV(A,b):
    M = np.concatenate((b,A),axis=1)
    mat = cdd.Matrix(M, number_type='float')
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    ext = poly.get_generators()
    ext = np.array(ext)
    return ext        