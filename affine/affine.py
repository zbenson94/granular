# ---------------------------------------------------------------------
# Function returns d2min
# ---------------------------------------------------------------------
# Importing external dependencies
# ---------------------------------------------------------------------
import numpy as np
from numpy import linalg as LA
from scipy import optimize
import granular.operations as op
# ---------------------------------------------------------------------
# Importing internal dependencies
#
# NO INTERNAL DEPENDENCIES NEEDED
#
# ---------------------------------------------------------------------
# Decomposes a deformation matrix into a rotation and symmetric matrix
# ---------------------------------------------------------------------
# Returns the collective rotations 
def s2min(dij0,dijf,R,forces=[],radius=0.25,beta=.0,doTwist=0):
    if (len(forces)==0):
        forces = np.ones(len(dij0[:,0]))
    # Assume the neighbors are present in both frames
    Rdij0 = op.vectorRotate(R,dij0)
    if(doTwist):
        ref = np.zeros((len(dij0[:,0]),3))
        ref[:,0] = 1
        ref[:,1] = 1
        ref[:,2] = 1
        xij0   = op.norm(np.cross(ref,dijf,axis=1))
        xijf   = op.norm(np.cross(xij0,op.vectorRotate(R,xij0),axis=1))
    else:
        xij0   = np.zeros(dij0.shape)
        xijf   = np.zeros(dij0.shape)
    # Final contact position
    yijf  = op.norm(Rdij0 - 2*op.vectorScale(op.vectorDot(Rdij0,dijf),dijf))
    yijf  = yijf + beta * xijf
    Y     = op.mkTensor(yijf,inv=1) 
    # Initial contact point transposed
    D     = op.mkTensor(dij0 + beta * xij0,inv=0)
    # Solution is the largest positive eigenvalue
    DtrY   = -np.sum(op.matrixProd(D,Y,sc=forces,tr=1),axis=0)
    val,vec   = np.linalg.eig(DtrY.astype('float32'))
    # Sort the eigenvectors
    idx = np.argsort(val)
    val = val[idx]
    vec = vec[:,idx]
    # Reduce to < pi rotation 
    idx = vec[0,:] < 0
    vec[:,idx] = - vec[:,idx]
    mag = np.sum(2 * forces) - 2 * val[3]
    return val,vec,len(dij0[:,0]),mag,DtrY



# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
def calcd2min(pos_j,pos_nbrs,quiet=1):
	
	dx_ti = pos_nbrs[:,0,0] - pos_j[0,0]
	dy_ti = pos_nbrs[:,1,0] - pos_j[1,0]
	dz_ti = pos_nbrs[:,2,0] - pos_j[2,0]

	dx_tf = pos_nbrs[:,0,1] - pos_j[0,1]
	dy_tf = pos_nbrs[:,1,1] - pos_j[1,1]
	dz_tf = pos_nbrs[:,2,1] - pos_j[2,1]

	ddxij = dx_tf - dx_ti
	ddyij = dy_tf - dy_ti
	ddzij = dz_tf - dz_ti

	contacts = len(dx_ti);
	def funct(x):
		return np.sum(
			np.square(ddxij - (x[0]*dx_ti + x[1]*dy_ti + x[2]*dz_ti)) + 
			np.square(ddyij - (x[3]*dx_ti + x[4]*dy_ti + x[5]*dz_ti)) + 
			np.square(ddzij - (x[6]*dx_ti + x[7]*dy_ti + x[8]*dz_ti))
			)
	if( not quiet):
		print('--------------------------------------')
		print('Neighbors Used: ' + str(contacts))
	
	xopt 	= optimize.fmin_bfgs(funct,x0=[1,0,0,0,1,0,0,0,1],disp=False)
	

	F,eVals = _decomp(xopt)

	return F,eVals,funct(xopt) / contacts




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
