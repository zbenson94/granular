#!/usr/bin/env python
# ---------------------------------------------------------------------
# Importing external dependencies
# ---------------------------------------------------------------------
import numpy as np
import numpy.linalg as LA
import granular.operations as op
# ---------------------------------------------------------------------



# Returns a rotation matrix between 2 coordinate systems
def getRotationMatrix(p0,p1):
    
    R = np.zeros((len(p0[:,0]),9))

    c0     = np.cross(p0[:,0:3],p0[:,3:6],axis=1)
    c1     = np.cross(p1[:,0:3],p1[:,3:6],axis=1)

    R[:,0] = np.sum(p1[:,0:3]*p0[:,0:3],axis=1)
    R[:,1] = np.sum(p1[:,0:3]*p0[:,3:6],axis=1)
    R[:,2] = np.sum(p1[:,0:3]*c0,axis=1)

    R[:,3] = np.sum(p1[:,3:6]*p0[:,0:3],axis=1)
    R[:,4] = np.sum(p1[:,3:6]*p0[:,3:6],axis=1)
    R[:,5] = np.sum(p1[:,3:6]*c0,axis=1)

    R[:,6] = np.sum(c1*p0[:,0:3],axis=1)
    R[:,7] = np.sum(c1*p0[:,3:6],axis=1)
    R[:,8] = np.sum(c1*c0,axis=1)

    return R 


def vectorRotate(R,d):
    Rd = np.zeros(d.shape)
    Rd[:,0] = np.sum(R[:,0:3]*d,axis=1)
    Rd[:,1] = np.sum(R[:,3:6]*d,axis=1)
    Rd[:,2] = np.sum(R[:,6:9]*d,axis=1)
    return Rd

def vectorScale(sc,v):

    vsc = np.zeros(v.shape)
    vsc[:,0] = sc * v[:,0]
    vsc[:,1] = sc * v[:,1]
    vsc[:,2] = sc * v[:,2]

    return vsc

# Not the greatest, but a matrix product
def matrixProd(A,B,sc=[],tr=1):

    AB = np.zeros(A.shape)
    if(len(sc) == 0):
       sc = np.ones((len(A[:,0,0]),1)) 
    for i in range(4):
        for j in range(4):
            if(tr==1):
                AB[:,i,j] = np.transpose(sc) * np.sum(A[:,:,i]*B[:,:,j],axis=1)
            else:
                AB[:,i,j] = np.transpose(sc) * np.sum(A[:,i,:]*B[:,:,j],axis=1)
    return AB

# Makes a tensor based on a vector
def mkTensor(v,inv=0):

    A = np.zeros((len(v[:,0]),4,4))

    A[:,0,1:] = -v
    A[:,1:,0] = v
    
    if(inv==0):
        A[:,1,2]  = v[:,2]
        A[:,1,3]  = -v[:,1]

        A[:,2,1]  = -v[:,2]
        A[:,2,3]  = v[:,0]

        A[:,3,1]  = v[:,1]
        A[:,3,2]  = -v[:,0]
    else:
        A[:,1,2]  = -v[:,2]
        A[:,1,3]  = v[:,1]

        A[:,2,1]  = v[:,2]
        A[:,2,3]  = -v[:,0]

        A[:,3,1]  = -v[:,1]
        A[:,3,2]  = v[:,0]
    return A 

# Creates a 3x3 cross product tensor
def crossProduct(A):
    Ac = np.zeros((len(A[:,0]),3,3))

    Ac[:,0,1] = - A[:,2] 
    Ac[:,0,2] =   A[:,1] 
    
    Ac[:,1,0] =   A[:,2] 
    Ac[:,1,2] = - A[:,0] 
    
    Ac[:,2,0] = - A[:,1] 
    Ac[:,2,1] =   A[:,0] 

    return Ac

# Returns outter product (xx^T)
def outerProduct(A,B):
    AB = np.zeros((len(A[:,0]),3,3))
    for i in range(3):
        for j in range(3):
            AB[:,i,j] = A[:,i] * B[:,j]

    return AB        

# Get the rotation axis
def getRotationAxis(R):
    axis = np.zeros((len(R[:,0]),3))
    axis[:,0]  = 0.5 * (R[:,7] - R[:,5])
    axis[:,1]  = 0.5 * (R[:,2] - R[:,6])
    axis[:,2]  = 0.5 * (R[:,3] - R[:,1])
    
    sintheta = LA.norm(axis,axis=1)

    return vectorScale(1./sintheta,axis)

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
