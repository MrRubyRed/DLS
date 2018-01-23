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

class Polytope_Tree(object):
    
    def __init__(self,state,policy,dynamics,pW,dW,dt):
        self.root = HPolytope(state,policy,dynamics,pW,dW,dt)
        poly_points = []
        self.find_all_dynamic_neighbors(self.root,policy,dynamics,pW,dW,dt,0,poly_points)
        #ponts = np.concatenate(poly_points,axis=0)
        #plt.scatter(ponts[:-1,0],ponts[:-1,1],c='b')
        #plt.scatter(ponts[-1,0],ponts[-1,1],c='r')
        #plt.show()
        #plt.pause(0.01)
        
    def find_all_dynamic_neighbors(self,node,policy,dynamics,pW,dW,dt,depth,poly_points):
        poly_points.append(node.interior_point)
        ponts = np.concatenate(poly_points,axis=0)
        plt.clf()
        plt.scatter(ponts[:-1,0],ponts[:-1,1],c='b')
        plt.scatter(ponts[-1,0],ponts[-1,1],c='r')
        plt.show()
        plt.pause(0.01)
        flag = False
        for i in range(len(node.exit_facets)):
            if node.exit_facets[i]:
                tmp_point = np.copy(node.points[i])
                while np.inner(node.A[i,:],tmp_point) <= node.b[i][0]:
                    tmp_point += 1e-8*node.A[i,:]
                    flag = True
                tmp_neighbor = node.check_if_in_union_BFS(tmp_point)
                if tmp_neighbor is None:
                    node.neighbors.append(HPolytope(tmp_point[None].T,policy,dynamics,pW,dW,dt))
                    node.neighbors[-1].neighbors.append(node)
                else:
                    node.neighbors.append(tmp_neighbor)
        
        node.neighbors_found = True
        
        for neighbor in node.neighbors:
            if not neighbor.neighbors_found:
                print("Starting New Search. Current Recursive Depth: " + str(depth) + " || Interior P=" + str(neighbor.interior_point) +" || Contains Rays = " + str(len(neighbor.R)>0) + " || " + str(flag))
                self.find_all_dynamic_neighbors(neighbor,policy,dynamics,pW,dW,dt,depth+1,poly_points)
        
        print("All neighbors found at a depth="+str(depth))
        
    def check_invariance(self,state,steps,policy,dynamics,pW,dW,dt):
        checks = []
        traj=[]
        new_state = np.copy(state)
        traj.append(new_state)
        node = self.root
        for i in range(steps):
            new_state = dynamics.step(new_state,policy.act(False,new_state)[0])[0][0]
            traj.append(new_state)
            node = node.check_if_in_union_BFS(state)
            if(node is None):
                checks.append(False)
            else:
                checks.append(True)
        traj = np.array(traj)
        plt.scatter(traj[:,0],traj[:,1],c='g')
        plt.pause(0.01)
        return checks
        
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
        self.points,self.exit_facets = self.find_facet_points_and_exit_facets_()
        self.points = np.concatenate(self.points)
        
        self.check = (np.abs(np.diagonal(np.matmul(self.A,self.points.T) - self.b)) < 1e-6).all()
        if not self.check: print(np.abs(np.diagonal(np.matmul(self.A,self.points.T) - self.b)))
        #print(str(self.A))
        #print(str(self.b))
        print(self.check)
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
            
        return points,exit_facets            
                
    def get_polytope(self,state,policy,dynamics,pW,dW,dt):
        list_Wp,list_bp = pW                                    #Unpack weightd and biases for the policy
        Ap,bp = self.get_region(list_Wp,list_bp,state)          #Get H-representation Polytope for state
        K = policy.get_Jacobian(state.T)[0][0]                  #Get K (from u = Kx + d) [Takes in np.arrat.shape = (1,n)] [returns np.array.spahe = (k,n,n)]
        d = policy.act(False,state.T[0])[0][None].T - np.matmul(K,state) #Get d (from u = Kx + d) [Takes in np.arrat.shape = (n,)] [returns (np.array.shape=(n,),float)]
        
        action = np.matmul(K,state)+d                           #Get action
        state_action = np.concatenate((state,action),axis=0)    #Concatenate state and action
        list_Wd,list_bd = dW                                    #Unpack weights and biases for the dynamics
        Ad,bd = self.get_region(list_Wd,list_bd,state_action)   #Get H-representation Polytope for state-action
        
        dyn_A,dyn_B = dynamics.get_Jacobian(state.T[0],action.T[0])       #Get dynamic matrices
        dyn_A = dyn_A[0]
        dyn_B = dyn_B[0]
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
        points = []
        for i in range(len(self.A)):
            c_ = self.A[i,None,:]
            c_n = c_/(np.linalg.norm(c_) + 1e-17)
            A_ = np.concatenate((self.A,-self.A[i,None,:]),axis=0)
            b_ = np.concatenate((self.b,-self.b[i,None,:]),axis=0)
            G = cvx.matrix(A_)
            h = cvx.matrix(b_)
            
            c = cvx.matrix(-np.matmul(c_,self.dyn_A)[0])
            sol = cvx.solvers.lp(c, G, h)
            
            if sol["status"] == "optimal":
                val = -sol["primal objective"]+np.matmul(c_,self.dyn_b)[0][0]
                x_s = np.array(sol["x"])
                
                if(val > 0):
                    exit_facets.append(True)
                else:
                    exit_facets.append(False)
            elif sol["status"] == "dual infeasible":
                print("UNBOUNDED POLYTOPE")
                exit_facets.append(True)
                sol_ = cvx.solvers.lp(-c, G, h)
                x_s = np.array(sol_["x"])
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
            mean_beta = np.mean(beta[inters])
            
            points.append(x_s.T + mean_beta*v)
                        
        return points,exit_facets
        
        
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
    
        
