# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 02:09:40 2017

@author: cusgadmin
"""
# DONE: Find ratio for polytope vs inner ellipsoid
# DONE: Find ratio for polytope vs random point hull 
# TODO: Find ratio for polytope vs charge separation point hull


import numpy as np
import cvxpy
import cdd
import time

# Projection
# c.T*(at + d) + b = 0
# a(c.T*t) + c.T*d + b = 0
# a* = - (b + c.T*d)/(c.T*t)
# Point = ta* + d

def polytope_sample(A,b,i_point): #We assume b <= 0
    alpha_list = []
    t = np.random.multivariate_normal(np.zeros(A.shape[1]),np.eye(A.shape[1]),size=(1,)).T
    for r in range(len(A)):
        numerator = (b[r,0] + np.matmul(A[r,:],i_point)[0])
        denominator = (np.matmul(A[r,:],t)[0])
        alpha = -numerator/denominator
        p = alpha*t + i_point
        #print(max((np.matmul(A,p) + b).T[0]))
        if((np.matmul(A,p) + b <= 1e-9).all()):
            alpha_list.append(alpha)
        if(len(alpha_list) == 2):
            break;

    if(len(alpha_list) < 2):
        print("Error! Less than two intersection points")
#        for r in range(len(A)):
#            numerator = (b[r,0] + np.matmul(A[r,:],i_point)[0])
#            denominator = (np.matmul(A[r,:],t)[0])
#            alpha = -numerator/denominator
#            p = alpha*t + i_point
#            print(max((np.matmul(A,p) + b).T[0]))
#            if((np.matmul(A,p) + b <= 1e-9).all()):
#                alpha_list.append(alpha)
#            if(len(alpha_list) == 2):
#                break;
        
    alpha_sorted = np.sort(alpha_list)
    alpha_final = np.random.uniform(alpha_sorted[0],alpha_sorted[1])
    
    return alpha_final*t + i_point

def ellipsoid_sample(B,d): #We assume b <= 0
    N = B.shape[1]
    t = np.random.multivariate_normal(np.zeros(N),np.eye(N),size=(1,)).T
    t = t/np.linalg.norm(t)
    alpha = np.random.uniform() 
    t = alpha*t
    
    return np.matmul(B,t) + d

#Note that it may happen that the origin is part of an unbounded polyhedron. Add more faces to avoid that.
def generate_random_polytope(dim,faces,scale=1.0,filt=False):
    if(faces < dim + 1):
        print("Error, need at least d+1 faces for a closed polytope")
        return None
    flag = True
    x = cvxpy.Variable(dim)
    while flag:
        A = scale*np.random.multivariate_normal(np.zeros(dim),np.eye(dim),size=(faces,))
        b = scale*np.random.uniform(-1,0,(faces,1))
        vertices = []
        flag = False
        for r in range(len(A)):
            objective = cvxpy.Maximize(-A[r,:]*x)
            constraints = [A*x + b <= 0]
            prob = cvxpy.Problem(objective,constraints)
            prob.solve()
            if prob.status == 'unbounded': #if all opposite facet directions are bounded, then the polytope is bounded
                flag = True
                break;
            vertices.append(np.array(x.value))
            
    if filt: #Remove redundant constraints
        x = cvxpy.Variable(dim)
        TorF = []
        for i in range(len(A)):
            objective = cvxpy.Minimize(-A[i,None,:]*x)
            constraints = [A*x + b <= 0]
            prob = cvxpy.Problem(objective,constraints)
            prob.solve()
            x_val = np.array(x.value)
            if(np.abs(np.matmul(A[i,None,:],x_val) + b[i]) <= 1e-5):
                TorF.append(True)
            else:
                TorF.append(False)
        A = A[np.array(TorF)]
        b = b[np.array(TorF)]
    
    return A,b,vertices
    
def generate_random_hullpoints(A,b,i_point,num_hullp = 100, filt=False): #We assume b <= 0
    hull_list = []
    for i in range(num_hullp/2):
        alpha_count = 0
        t = np.random.multivariate_normal(np.zeros(A.shape[1]),np.eye(A.shape[1]),size=(1,)).T
        for r in range(len(A)):
            numerator = (b[r,0] + np.matmul(A[r,:],i_point)[0])
            denominator = (np.matmul(A[r,:],t)[0])
            alpha = -numerator/denominator
            p = alpha*t + i_point
            #print(max((np.matmul(A,p) + b).T[0]))
            if((np.matmul(A,p) + b <= 1e-9).all()):
                hull_list.append(p)
                alpha_count += 1
                
            if(alpha_count == 2):
                break;
                
    if filt:   
        B = cvxpy.Symmetric(A.shape[1])
        d = cvxpy.Variable(A.shape[1])
        objective = cvxpy.Minimize(-cvxpy.log_det(B))
        constraints = [cvxpy.norm(B*v + d) <= 1.0 for v in hull_list]
        constraints.append(B >> 0)
        prob = cvxpy.Problem(objective,constraints)
        prob.solve()
        B = np.array(B.value)
        d = np.array(d.value)
        
        hull_list_f = [h for h in hull_list if (np.abs(1.0 - np.linalg.norm(np.matmul(B,h)+d)) <= 1e-7)]

    return hull_list,hull_list_f      

def generate_random_vertices(A,b,num_vertices = 100): #We assume b <= 0
    vertices_list = []
    x = cvxpy.Variable(A.shape[1])
    
    for i in range(num_vertices):
        
        t = np.random.multivariate_normal(np.zeros(A.shape[1]),np.eye(A.shape[1]),size=(1,))
        objective = cvxpy.Maximize(t*x)
        constraints = [A*x + b <= 0]
        prob = cvxpy.Problem(objective,constraints)
        prob.solve()
        vertices_list.append(np.array(x.value))

    return vertices_list     

def estimate_volume_ratio_chull(A,b,i_point,vertex_list,ratio_num=1000,verbose=False):
    
    vertex_mat = np.concatenate(vertex_list,axis=1)
    lam = cvxpy.Variable(len(vertex_list),1)
    ones_vec = np.ones((1,len(vertex_list)))

    inout = 0.0    
    
    for n in range(ratio_num):
        p = polytope_sample(A,b,i_point)
        objective = cvxpy.Maximize(0)
        constraints = [vertex_mat*lam == p]
        constraints.append(ones_vec*lam == 1.0)
        constraints.append(lam >= 0.0)
        prob = cvxpy.Problem(objective,constraints)
        prob.solve()
        if prob.status == 'optimal':
            inout += 1.0
        if (n%(ratio_num/10) == 0) and verbose:
            print(inout/ratio_num)

    return inout/ratio_num

def get_maxvol_insc_ellipsoid(A,b):
    N = A.shape[1]
    B = cvxpy.Symmetric(N)
    d = cvxpy.Variable(N)

    objective = cvxpy.Maximize(cvxpy.log_det(B))
    constraints = [cvxpy.norm(B*A[i,:,None]) + A[i,:,None].T*d <= -b[i] for i in range(len(b))]
    constraints.append(B >> 0)
    prob = cvxpy.Problem(objective,constraints)
    prob.solve()

    B = np.array(B.value)
    d = np.array(d.value)

    return B,d

def get_minvol_outer_ellipsoid(V,sequential=False):
    N = V.shape[1]
    B = cvxpy.Symmetric(N)
    d = cvxpy.Variable(N,1)
    if not sequential:
        objective = cvxpy.Minimize(-cvxpy.log_det(B))    
        constraints = [cvxpy.norm(B*V[[i],:].T + d) <= 1 for i in range(len(V))]
        constraints.append(B >> 0)
        prob = cvxpy.Problem(objective,constraints)
        prob.solve()    
    
        B = np.array(B.value)
        d = np.array(d.value)
    
        return B,d
    else:
        upper = np.argmax(V,axis=0)
        lower = np.argmin(V,axis=0)
        rem = np.concatenate((upper,lower))
        upper_bound_points = np.array([V[k,:] for k in upper])
        lower_bound_points = np.array([V[k,:] for k in lower])
        V = np.delete(V,rem,0)
        V_ = np.concatenate((upper_bound_points,lower_bound_points),axis=0)
        
        objective = cvxpy.Minimize(-cvxpy.log_det(B))    
        constraints = [cvxpy.norm(B*V_[[i],:].T + d) <= 1 for i in range(len(V_))]
        constraints.append(B >> 0)
        prob = cvxpy.Problem(objective,constraints)
        prob.solve()    
    
        B = np.array(B.value)
        d = np.array(d.value) 
        
        outside = np.linalg.norm(np.matmul(np.linalg.inv(B),V.T-d),axis=0) 
        
        while (outside > 1.0).any():
            B = cvxpy.Symmetric(N)
            d = cvxpy.Variable(N,1)
            constraints = []
            
            indx = np.argsort(outside)[-N:]
            new_points = np.array([V[k,:] for k in indx])
            V_ = np.concatenate((V_,new_points),axis=0)
            V = np.delete(V,indx,0)
        
            objective = cvxpy.Minimize(-cvxpy.log_det(B))    
            constraints = [cvxpy.norm(B*V_[[i],:].T + d) <= 1 for i in range(len(V_))]
            constraints.append(B >> 0)
            prob = cvxpy.Problem(objective,constraints)
            prob.solve()    
        
            B = np.array(B.value)
            d = np.array(d.value) 
    
            outside = np.linalg.norm(np.matmul(np.linalg.inv(B),V.T-d),axis=0) 
            print(np.sum(outside > 1.0))
    
def polytiopic_ellipsoidal_partition(A,b,list_ellips,num_part):
    
    B,d = get_maxvol_insc_ellipsoid(A,b)   
    
    if num_part == 0:
        list_ellips.append((B,d))
        return
    else:
        _,eigvec = np.linalg.eig(B)
        max_vec = eigvec[:,[0]].T
        polytiopic_ellipsoidal_partition(np.concatenate((A,max_vec),axis=0),
                                         np.concatenate((b,np.matmul(-max_vec,d)),axis=0),
                                         list_ellips,
                                         num_part=num_part-1)
        polytiopic_ellipsoidal_partition(np.concatenate((A,-max_vec),axis=0),
                                         np.concatenate((b,np.matmul(max_vec,d)),axis=0),
                                         list_ellips,
                                         num_part=num_part-1)        

def estimate_volume_ratio_partition(A, b, list_ellips, ratio_num=1000):
    
    N = A.shape[1]
    
    inout = 0.0 #number of sampled points which fall both inside the polytope and the ellipse
    
    for n in range(ratio_num):
        ind = np.random.randint(len(list_ellips))
        B,d = list_ellips[ind]
        B = N*B
        p = ellipsoid_sample(B,d)
        
        if (np.matmul(A,p) + b <= 0).all():
            inout += 1.0
            
    return (ratio_num-inout)/ratio_num
        

def estimate_volume_ratio_ellipsoid(A,b,i_point,ratio_num=1000, LJ=False, verbose=False):
    
    N = A.shape[1]
    B = cvxpy.Symmetric(N)
    d = cvxpy.Variable(N)

    objective = cvxpy.Maximize(cvxpy.log_det(B))
    constraints = [cvxpy.norm(B*A[i,:,None]) + A[i,:,None].T*d <= -b[i] for i in range(len(b))]
    constraints.append(B >> 0)
    prob = cvxpy.Problem(objective,constraints)
    prob.solve()#solver=cvxpy.CVXOPT, verbose=True, kktsolver='robust')    
    print(prob.status)

    B = np.array(B.value)
    d = np.array(d.value)
    if LJ:
        B = N*B
    
    B_inv = np.linalg.inv(B)

    inout = 0.0    
    
    for n in range(ratio_num):
        p = polytope_sample(A,b,i_point)
        if(np.linalg.norm(np.matmul(B_inv,(p-d))) < 1.0):
            inout += 1
        
        if (n%(ratio_num/10) == 0) and verbose:
            print(inout/ratio_num)

    return inout/ratio_num, B, d

def estimate_volume_ratio_sphere(A,b,i_point,ratio_num=1000,verbose=False):
    
    r = cvxpy.Variable()
    xc = cvxpy.Variable(A.shape[1])
    
    norm_a = np.linalg.norm(A,axis=1)

    objective = cvxpy.Maximize(r)
    constraints = [A*xc + r*norm_a <= -b]
    prob = cvxpy.Problem(objective,constraints)
    prob.solve()    
    print(prob.status)

    xc = np.array(xc.value)
    r = np.array(r.value)

    inout = 0.0    
    
    for n in range(ratio_num):
        p = polytope_sample(A,b,i_point)
        if(np.linalg.norm(p-xc) < r):
            inout += 1
        
        if (n%(ratio_num/10) == 0) and verbose:
            print(inout/ratio_num)

    return inout/ratio_num

def convert_HtoV(A,b):
    M = np.concatenate((-b,A),axis=1)
    mat = cdd.Matrix(M, number_type='float')
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    ext = poly.get_generators()
    ext = np.array(ext)
    return ext       

def get_face_points(A,b):
    points = []
    x_ac = cvxpy.Variable(A.shape[1])
    for i in range(len(A)):
        A_tmp = np.concatenate((A,-A[i,None,:]),axis=0)
        b_tmp = np.concatenate((b,-b[i,None,:]-1e-5),axis=0)
        objective = cvxpy.Minimize(-cvxpy.sum_entries(cvxpy.log(-A_tmp*x_ac - b_tmp)))
        constraints = []
        prob = cvxpy.Problem(objective,constraints)
        prob.solve()#solver=cvxpy.SCS)#solver=cvxpy.CVXOPT, verbose=True, kktsolver='robust')    
        print(prob.status) 
        points.append(np.array(x_ac.value))
    return points
    
A,b,_ = generate_random_polytope(10,30,scale=2.0,filt=True)
i_point = np.zeros((A.shape[1],1))


#points = get_face_points(A,b)

#t = time.time()
vertices = convert_HtoV(A,b)
vertices = vertices[:,1:]
print(vertices)
#t = time.time() - t
#print(len(vertices))
#print(t)
    #list_ellips = []
    #polytiopic_ellipsoidal_partition(A, b, list_ellips=list_ellips, num_part=0)
    #ratio1 = estimate_volume_ratio_partition(A,b,list_ellips,ratio_num=10000)
#list_ellips = []
#polytiopic_ellipsoidal_partition(A, b, list_ellips=list_ellips, num_part=4)
#ratio2 = estimate_volume_ratio_partition(A,b,list_ellips,ratio_num=10000)
B,d = get_minvol_outer_ellipsoid(vertices,sequential=True)
list_ellips = [(B,d)]
ratio2 = estimate_volume_ratio_partition(A,b,list_ellips,ratio_num=10000)
#ratio1,B1,d1 = estimate_volume_ratio_ellipsoid(A,b,i_point,ratio_num=1000,LJ=False,verbose=False)
#ratio2,B2,d2 = estimate_volume_ratio_ellipsoid(A,b,i_point,ratio_num=1000,LJ=True,verbose=False)
##ratio = estimate_volume_ratio_sphere(A,b,i_point,ratio_num=1000,verbose=False)
##vertex_list,vertex_list_f = generate_random_hullpoints(A,b,i_point,num_hullp=1000,filt=True)
##vertex_list = generate_random_vertices(A,b,num_vertices=500)

#vertex_mat = np.concatenate(vertex_list,axis=1)



#ratio = estimate_volume_ratio_chull(A,b,i_point,vertex_list,ratio_num=1000,verbose=False)
#print(ratio1)
#ratio = estimate_volume_ratio_chull(A,b,i_point,vertex_list_f,ratio_num=1000,verbose=False)
print(ratio2)