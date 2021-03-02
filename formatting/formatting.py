# Author: ZAB
# Date:   03-02-2021
# ---------------------------------------------------------------------
# External dependencies
# ---------------------------------------------------------------------
import pandas as pd
import numpy as np 
import scipy as sp
# ---------------------------------------------------------------------
# Local dependencies
# ---------------------------------------------------------------------
import ..dataloading as ssio


# ---------------------------------------------------------------------
# Function will produce an contact list, the positions
# and orientations of all the grains in a .pkl file
# ---------------------------------------------------------------------
def mkdtf(fname,outfname,N=20000,r_sq=0.25):


	xyz = ssio.read_SS(fname)[0][0:N,:]
	ori = ssio.read_SS_ori(fname + '.ori')[0:N,:]


	dtf = pd.DataFrame()


	nbrs = {f:[] for f in range(N)}


	IDs = np.arange(N,dtype='int')

	idx = np.zeros(N,dtype='bool')

	for n in range(N):

		dr = np.sum(np.square(xyz[n,:] - xyz[n+1:,:]),axis=1) <= r_sq


		idx[0:n+1] = False
		idx[n+1:]  = dr

		nbrs[n] = list(set(nbrs[n]).union(IDs[idx]))

		for j in IDs[idx]:

			nbrs[j] = list(set(nbrs[j]).union([n]))


	dtf["neighbors"] = nbrs.values()
	dtf["xyz"]       = xyz
	dtf["p1"]        = ori[:,0:3]
	dtf["p2"]        = ori[:,3:6]

	dtf.to_pickle(outfname + '.pkl')


# Returns an adjacency matrix given an edge list
# ---------------------------------------------------------------------
def getAdj(dtf,binary=True,force=True,d=0.5):

	N = len(dtf)

	if(binary):
		adj = np.zeros((N,N),dtype='float32')

		for n in range(N):

			adj[n,dtf["neighbors"].values[n]] = 1
			adj[dtf["neighbors"].values[n],n] = 1

			
	else:
		adj = np.zeros((N,N),dtype='float32')

		for n in range(N):
			nbrs = dtf["neighors"].values[n]

			dr   = np.sum(np.square(dtf["xyz"].values[nbrs,:] - dtf["xyz"].values[n,:]))

			if(force):
				adj[n,nbrs] = d - np.sqrt(dr)
				adj[nbrs,n] = d - np.sqrt(dr)
			else:
				adj[n,nbrs] = dr
				adj[nbrs,n] = dr


	return adj








