#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
ssio.py
utility for reading and writing ss files
author: Ron Ballouz (with mods by DCR and JVD)
date: 11/22/2015 (original version)
functions: read_SS(), write_SS() -- see below for invocations
to use: import ssio
'''

import xdrlib
import numpy as np
import networkx as nx
import os

class ReadError(Exception):
    '''Passes argument to Exception class'''
    def __init__(self, arg):
        super().__init__(arg)


class WriteError(Exception):
    '''Passes argument to Exception class'''
    def __init__(self, arg):
        super().__init__(arg)

lu 		= 6.68458e-14
tu 		= 1.99102e-07

##############################################################################
#
# read_SS()
# author       : Ron Ballouz
# date         : 11/22/2015
# Description  : function to read ss files
# Invocation   : [list,] numpy_array [or list] =
#                read_SS ( filename (string) [ , output_type (string) ]
#                [ , unpack=False ] )
# Input        : filename (required), output_type (optional)
# Returns      :
#   1. optionally the header info as a list: [time, N, magic number]
#   2. either a numpy 14 x N array of the data or a list if unpack=True
#      (the data format is order number, original index, mass, radius,
#      position, velocity, spin, and color in both cases)
#
##############################################################################

def read_SS(file, headout='no', unpack=False):
    if type(file) is not str:
        raise ReadError('The filename argument must be a string')
    if type(headout) is not str:
        raise ReadError('The output_type argument must be a string')

    f = open(file, 'rb').read()
    data = xdrlib.Unpacker(f)

    # Header data
    time = data.unpack_double()
    N = data.unpack_int()
    iMagicNumber = data.unpack_int()

    if N < 1:
        raise ReadError('No particles found')

    # Initialize data
    Index 	= np.zeros(N, dtype='int32')
    mass 	= np.zeros(N, dtype='float64')
    rad 	= np.zeros(N, dtype='float64')
    pos 	= np.zeros((N, 3), dtype='float64')
    vel 	= np.zeros((N, 3), dtype='float64')
    spin 	= np.zeros((N, 3), dtype='float64')
    color 	= np.zeros(N, dtype='int16')

    # Actual data (see ssio.h for data order)
    if iMagicNumber == -1:
        for i in np.arange(N):
            mass[i] 	= data.unpack_double()
            rad[i] 		= data.unpack_double()
            pos[i,0] 	= data.unpack_double()
            pos[i,1] 	= data.unpack_double()
            pos[i,2] 	= data.unpack_double()
            vel[i,0] 	= data.unpack_double()
            vel[i,1] 	= data.unpack_double()
            vel[i,1] 	= data.unpack_double()
            spin[i,0] 	= data.unpack_double()
            spin[i,1] 	= data.unpack_double()
            spin[i,2] 	= data.unpack_double()
            color[i] 	= data.unpack_int()
            Index[i] 	= data.unpack_int()
    elif iMagicNumber == -10101:
        for i in np.arange(N):
            mass[i] 	= data.unpack_float()
            rad[i] 		= data.unpack_float()
            pos[0, i] 	= data.unpack_float()
            pos[1, i] 	= data.unpack_float()
            pos[2, i] 	= data.unpack_float()
            color[i] 	= data.unpack_int()
            Index[i] 	= data.unpack_int()
    else:
        raise ReadError('Invalid magic number (corrupt file?)')

    data.done()

    return pos / lu ,vel / lu * tu ,spin * tu


def read_SS_ori(file, headout='no', unpack=False):
	if type(file) is not str:
		raise ReadError('The filename argument must be a string')
	if type(headout) is not str:
		raise ReadError('The output_type argument must be a string')

	f = open(file, 'rb').read()
	data = xdrlib.Unpacker(f)

    # Header data
	time = data.unpack_double()
	N = data.unpack_int()
	iMagicNumber = data.unpack_int()

	if N < 1:
		raise ReadError('No particles found')

    # Initialize data
	ori=np.zeros((N,6),dtype='float64')
    # Actual data (see ssio.h for data order)
	for i in np.arange(N):
		ori[i,0] = data.unpack_double()
		ori[i,1] = data.unpack_double()
		ori[i,2] = data.unpack_double()
		ori[i,3] = data.unpack_double()
		ori[i,4] = data.unpack_double()
		ori[i,5] = data.unpack_double()

	data.done()

	return ori

##############################################################################
#
# write_SS()
# author       : Ron Ballouz
# date         : 11/24/2015
# Description  : function to write ss files
# Invocation   : write_SS ( py_data (14xN numpy array) , filename (string),
#                           [timestamp (double)], [verbose (string)] )
# Input        : py_data (required), filename (required),
#                timestamp (optional, default 0),
#                verbose output (optional, default n)
# Returns      :
#
##############################################################################

def write_SS(py_data, filename, time=0, verbose='n'):
    if py_data.shape[0] != 14:
        raise WriteError('Invalid numpy array size')
    if type(filename) is not str:
        raise WriteError('The filename argument must be a string')
    if type(verbose) is not str:
        raise WriteError('The verbose option needs to be a string. '
                         'Use "v" for verbose output')
    if filename.split('.')[-1] == 'r':
        if verbose == 'v':
            print('Assuming data to be packed is reduced')
        iMagicNumber = -10101
    else:
        if verbose == 'v':
            print('Assuming data to be packed is full output')
        iMagicNumber = -1

    N = py_data.shape[1]
    if N < 1:
        raise WriteError('No particle data found')

    data = xdrlib.Packer()
    # Header Data
    data.pack_double(time)
    data.pack_int(N)
    data.pack_int(iMagicNumber)

    # Actual Data
    if iMagicNumber == -1:
        for i in np.arange(N):
            data.pack_double(py_data[2, i])
            data.pack_double(py_data[3, i])
            data.pack_double(py_data[4, i])
            data.pack_double(py_data[5, i])
            data.pack_double(py_data[6, i])
            data.pack_double(py_data[7, i])
            data.pack_double(py_data[8, i])
            data.pack_double(py_data[9, i])
            data.pack_double(py_data[10, i])
            data.pack_double(py_data[11, i])
            data.pack_double(py_data[12, i])
            data.pack_int(int(py_data[13, i]))
            data.pack_int(int(py_data[1, i]))
    elif iMagicNumber == -10101:
        for i in np.arange(N):
            data.pack_float(py_data[2, i])
            data.pack_float(py_data[3, i])
            data.pack_float(py_data[4, i])
            data.pack_float(py_data[5, i])
            data.pack_float(py_data[6, i])
            data.pack_int(int(py_data[13, i]))
            data.pack_int(int(py_data[1, i]))

    # Packing Data to return
    f = open(filename, 'wb')
    f.write(data.get_buffer())
    f.close()


def importData(fname,do_gpickle=0):

    if(os.path.isfile(fname)):
        if(do_gpickle):
            output = nx.read_gpickle(fname)
        else:
            output = np.load(fname)
        loaded = 1
    else:
        output = []
        loaded = 0 

    return output,loaded



