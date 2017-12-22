import numpy as np
import cvxpy

E1 = np.array([[-1.0,0.0,0.0],[-1.0,0.0,-1.0]])
E2 = np.array([[0.0,0.0],[-1.0,0.0]])
E3 = np.array([[0.0,0.0],[1.0,0.0]])
E4 = np.array([[1.0,0.0,-1.0],[1.0,0.0,0.0]])


A1 = np.array([[0.9,-0.1,0.0],[0.1,1,-0.02],[0.0,0.0,1.0]])
A2 = np.array([[1.0,-0.02],[0.02,0.9]])
A3 = np.array([[1.0,-0.02],[0.02,0.9]])
A4 = np.array([[0.9,-0.1,0.0],[0.1,1,0.02],[0.0,0.0,1.0]])

R = np.concatenate((np.eye(A2.shape[0]),np.zeros((A2.shape[0],1))),axis=1)
L = R.T

E2_ = np.matmul(E2,R)
E3_ = np.matmul(E3,R)
A2_ = np.matmul(L,np.matmul(A2,R))
A3_ = np.matmul(L,np.matmul(A3,R))

I2 = np.eye(2)
I3 = np.eye(3)

P1 = cvxpy.Symmetric(3)
P2 = cvxpy.Symmetric(2)
P3 = cvxpy.Symmetric(2)
P4 = cvxpy.Symmetric(3)

U1 = cvxpy.Symmetric(2)
U2 = cvxpy.Symmetric(2)
U3 = cvxpy.Symmetric(2)
U4 = cvxpy.Symmetric(2)

W1 = cvxpy.Symmetric(2)
W2 = cvxpy.Symmetric(2)
W3 = cvxpy.Symmetric(2)
W4 = cvxpy.Symmetric(2)

Q12 = cvxpy.Symmetric(2)
Q21 = cvxpy.Symmetric(2)
Q23 = cvxpy.Symmetric(2)
Q32 = cvxpy.Symmetric(2)
Q34 = cvxpy.Symmetric(2)
Q43 = cvxpy.Symmetric(2)

c=20.0
constraints = []
constraints.append(c*I2 << P2 - (E2.T)*U2*E2)
constraints.append(c*I2 << P3 - (E3.T)*U3*E3)
constraints.append((A2.T)*P2*A2 - P2 + (E2.T)*W2*E2 << c*I2) 
constraints.append((A3.T)*P3*A3 - P3 + (E3.T)*W3*E3 << c*I2)
constraints.append(c*I3 << P1 - (E1.T)*U1*E1)
constraints.append(c*I3 << P4 - (E4.T)*U4*E4)
constraints.append((A1.T)*P1*A1 - P1 + (E1.T)*W1*E1 << c*I3) 
constraints.append((A4.T)*P4*A4 - P4 + (E4.T)*W4*E4 << c*I3)
constraints.append((A2.T)*P3*A2 - P2 + (E2.T)*Q23*E2 << c*I2)
constraints.append((A3.T)*P2*A3 - P3 + (E3.T)*Q32*E3 << c*I2)
constraints.append((A1.T)*(L*P2*R)*A1 - P1 + (E1.T)*Q12*E1 << c*I3)
constraints.append((A4.T)*(L*P3*R)*A4 - P4 + (E4.T)*Q43*E4 << c*I3)
constraints.append(A2_.T*P1*A2_ - L*P2*R + E2_.T*Q21*E2_ << c*I3)
constraints.append(A3_.T*P4*A3_ - L*P3*R + E3_.T*Q34*E3_ << c*I3)
constraints.append(U1 > 0)
constraints.append(U2 > 0)
constraints.append(U3 > 0)
constraints.append(U4 > 0)
constraints.append(W1 > 0)
constraints.append(W2 > 0)
constraints.append(W3 > 0)
constraints.append(W4 > 0)
constraints.append(Q12 > 0)
constraints.append(Q21 > 0)
constraints.append(Q23 > 0)
constraints.append(Q32 > 0)
constraints.append(Q34 > 0)
constraints.append(Q43 > 0)

obj = cvxpy.Minimize(0)
Prob = cvxpy.Problem(obj,constraints)
Prob.solve()