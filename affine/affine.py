# ---------------------------------------------------------------------
# Function returns d2min
# ---------------------------------------------------------------------
# Importing external dependencies
# ---------------------------------------------------------------------
import numpy as np
from numpy import linalg as LA
from scipy import optimize


# ---------------------------------------------------------------------
# Importing internal dependencies
# ---------------------------------------------------------------------
import granular.operations as op


# ---------------------------------------------------------------------
# Returns the minimized least-square sliding between contacts
# ---------------------------------------------------------------------
def s2min(dij0,dijf,R,forces=[],radius=0.25):
    if (len(forces)==0):
        forces = np.ones(len(dij0[:,0]))
    # Assume the neighbors are present in both frames
    Rdij0 = op.vectorRotate(R,dij0)
    # Final contact position
    yijf  = op.norm(Rdij0 - 2*op.vectorScale(np.sum(Rdij0*dijf,axis=1),dijf))

    Y     = op.mkTensor(yijf,inv=1) 
    # Initial contact point transposed
    D     = op.mkTensor(dij0,inv=0)
    
    # Solution is the largest positive eigenvalue
    DtrY      = - np.sum(op.matrixProd(D,Y,sc=forces,tr=1),axis=0)
    val,vec   =   np.linalg.eig(DtrY.astype('float32'))
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
# Calculates optimum rotation that minimizes dissipation
# ---------------------------------------------------------------------
def work(dij0,dijf,R,forces,x=[]):
    
    
    Rdij0    = op.vectorRotate(R,dij0)
    
    # Reflect that point onto the correct axis
    MRdij0   = -(Rdij0 - 2 * op.vectorScale(np.sum(Rdij0*dijf,axis=1),dijf))
    
    yij      = np.cross(np.cross(dij0,MRdij0,axis=1),dij0,axis=1)
    
    s2min    = aff.s2min(dij0,dijf,R,forces)[1][:,3]
    
    
    sintheta     = LA.norm(s2min[1:])
    
    # Initial conditions for the work calculation
    initcond = s2min[1:] / sintheta * np.sin(2*np.arcsin(sintheta))

    
    def equation(x_fit):
           
        return np.sum(forces*np.sqrt(
                        np.square((x_fit[1]*dij0[:,2]-x_fit[2]*dij0[:,1]) - yij[:,0]) + 
                        np.square((x_fit[2]*dij0[:,0]-x_fit[0]*dij0[:,2]) - yij[:,1]) + 
                        np.square((x_fit[0]*dij0[:,1]-x_fit[1]*dij0[:,0]) - yij[:,2])
                    ))


    if(len(x)==0):
        return sp.optimize.minimize(equation,initcond,method='bfgs')
    else:
        return equation(x)

# ---------------------------------------------------------------------
# Collective motion calculation
# ---------------------------------------------------------------------
def calcd2min(dij0,dijf,quiet=1):
	
	def funct(x):
		return np.sum(
            np.square(dijf[:,0] - (x[0]*dij0[:,0] + x[1]*dij0[:,1] + x[2]*dij0[:,2])) + 
            np.square(dijf[:,1] - (x[3]*dij0[:,0] + x[4]*dij0[:,1] + x[5]*dij0[:,2])) + 
            np.square(dijf[:,2] - (x[6]*dij0[:,0] + x[7]*dij0[:,1] + x[8]*dij0[:,2])) + 
			)

	if( not quiet):
		print('--------------------------------------')
		print('Neighbors Used: ' + str(contacts))
	
	xopt 	= optimize.fmin_bfgs(funct,x0=[1,0,0,0,1,0,0,0,1],disp=False)
	
    # Returns the matrix, the minimum vlaue, and the neighbors used
	return xopt,funct(xopt),len(dijf[:,1])
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
