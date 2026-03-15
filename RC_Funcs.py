"""
Created on Feb  27.02.2026

@author: Dr. Manish Yadav
Email: manish.yadav@tu-berlin.de

This code contains the functions for the article "Emergent E-I Structure in Performance-Evolved Reservoir Networks of
Neuronal Population Dynamics" by Manish Yadav. The main functions are:
- rect_tanh: Rectified hyperbolic tangent activation function for the reservoir neurons.
- f: General activation function for the reservoir neurons, which can be scaled by g_scale.
- Reservoir: The main function to run the reservoir dynamics given the network, inputs, and parameters.
- RandNetGenerator: Generates a random Erdős–Rényi (ER) network with a specified average degree and spectral radius.
- RandNetTestGenerator: Generates a random ER network and checks its spectral radius to ensure it's suitable for RC training.
- InpOut_Init_Gen: Generates random input and output node indices for the reservoir based on the specified percentages.
- GNet_SpectralRadius: Rescales the reservoir network to have a desired spectral radius.
- RC_Train: Trains the RC model by running the reservoir and performing ridge regression to find the output weights.
- Generate_Winps: Generates the input weight matrix for the reservoir based on the input node indices.
- RC: The main function to run the RC model, which includes training and testing/prediction phases, and returns the predictions and performance metrics.
- Ridge_Regression: Performs ridge regression to compute the output weights from the reservoir states and target outputs.
- MSE: Computes the mean squared error between two arrays.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot
import copy
import networkx as nx
import time
from timeit import default_timer as timer
import os
import pickle
from scipy.integrate import solve_ivp
import Tasks as Task

from RC_Funcs import *
from PDNE_Functions import *

def sigmoid(x, a=4, theta=0.5):
    return 1/(1 + np.exp(-a*(x-theta)))

def neuron(x):
    return 1/(1+np.exp(-5*(x-0.2)))

def rect_tanh(x):
    return np.maximum(0, np.tanh(x))

def f(g_scale, x):
    return np.tanh(np.array(g_scale) * x)
    
### Reservoir functions#########################################################################################

def Reservoir(GNet, Init, Inps, Winps, N_I, N, alpha, g_scale, ResTrans):
    Nodes_res = GNet.shape[0]; 
    batch, Npts_U = Inps.shape[0], Inps.shape[2]

    R = np.zeros([N, batch, Npts_U])
    
    for b in range(batch):
        R[:,b,0] = Init[:]
        ####time loop
        for t in range(0, Npts_U-1):  
            Inp_term=0
            for i in range(N_I):
                Inp_term +=  Winps[i]*Inps[b,i,t]    
            R[:,b,t+1] = (1-alpha)*np.asarray(R[:,b,t]) + alpha*f(g_scale, np.dot(GNet,R[:,b,t]) + Inp_term)
            
    ###Drop Transients from each batch
    R_trans = R[:,:,ResTrans:]

    return R_trans

def RandNetGenerator(K,n,eig_rho):
    # # ER network n*n and its radius is eig_rho
    prob = K/(n)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            b = np.random.random()
            if (i != j) and (b < prob):
                W[i, j] = np.random.random()

    rad = max(abs(np.linalg.eigvals(W)))
    W_reservoirr = W*(eig_rho/rad)  
    return W_reservoirr

def RandNetTestGenerator(Ks, Na, eig_rhos):
    ###avoid these errors: "RuntimeWarning: divide by zero encountered in true_divide" and "RuntimeWarning: invalid value encountered in true_divide" when the spectral radius is 0 or very close to 0, which can cause problems in the RC training. So, we will regenerate the seed network until we get a good one with spectral radius less than 1.
    ###also look for this error "Array must not contain infs or NaNs"
    while True:
        GNet = RandNetGenerator(Ks, Na, eig_rhos)
        Spectral_radius_Gn = max(abs(np.linalg.eigvals(GNet)))
        print(f"Generated a seed network with spectral radius: {Spectral_radius_Gn}")
        print(f"Is the spectral radius less than 1? {'Yes' if Spectral_radius_Gn < 1 else 'No'}")
        if Spectral_radius_Gn < 1 and not np.isnan(Spectral_radius_Gn) and not np.isinf(Spectral_radius_Gn):
            print(f"Generated a good seed network with spectral radius: {Spectral_radius_Gn}")
            break
    return GNet

def InpOut_Init_Gen(Net_Init, P_inp, P_out, N_I, N_O):
    InpsNodes_init = list([[]])
    OutsNodes_init = list([[],[]])
    for i in range(N_I):
        ###random selecting 50% of the initial nodes as input nodes
        InpsNodes_init[i] = list(np.random.choice(Net_Init.shape[0], size=int(P_inp*Net_Init.shape[0]), replace=False))
    for i in range(N_O):
        ###random selecting 50% of the initial nodes as output nodes
        OutsNodes_init[i] = list(np.random.choice(Net_Init.shape[0], size=int(P_out*Net_Init.shape[0]), replace=False))

    # ###convert to list of lists of node indices for each input and output
    InpsNodes_init = [list(set(InpsNodes_init[i])) for i in range(N_I)]
    OutsNodes_init = [list(set(OutsNodes_init[i])) for i in range(N_O)]
    # ##print lens of input and output node lists
    print('Initial Input Nodes:', len(InpsNodes_init[0]))
    print('Initial Output Nodes:', len(OutsNodes_init[0]), len(OutsNodes_init[1]))
    return InpsNodes_init, OutsNodes_init

def GNet_SpectralRadius(Gn, Spectral_radius):
    ### Rescaling to a desired spectral radius 
    Spectral_radius_Gn = max(abs(np.linalg.eigvals(Gn)))
    ResMat = Gn*Spectral_radius/Spectral_radius_Gn
    return ResMat

def RC_Train(G, Spectral_radius, alpha, g_scale, N_I, N_O, Winps, OutsNodes, Inps, Outs, Trans, beta):
    ### G to Matrix
    GNet = nx.to_numpy_array(G)
    N = G.number_of_nodes()

    GNet = GNet_SpectralRadius(GNet, Spectral_radius)

    ### Reservoir run
    Init = np.zeros(N) 
    Res_3D = Reservoir(GNet, Init, Inps, Winps, N_I, N, alpha, g_scale, Trans)

    ### Training
    W_outs = Ridge_Regression(Res_3D.reshape(N, -1), beta, Outs[:,:,Trans:], N_O, OutsNodes)
    return GNet, Init, N, W_outs


def Generate_Winps(N_I, N, InpsNodes):
    Winps = np.zeros((N_I, N))
    Winps[0, InpsNodes[0]] = 0.8   
    return Winps

def RC(G, Spectral_radius, alpha, g_scale, N_I, N_O, InpsNodes, OutsNodes, Inps, Outs, Inps_test, Outs_test, Trans, RC_reps, beta, Return=0):
    ### The shape of each Predictions are (batch, N_O, timesteps). 
    # So, the shape of Outs_predict and Outs_test_pred should be (RC_reps, batch, N_O, timesteps)
    Outs_predict=np.zeros((RC_reps, Outs.shape[0], N_O, Outs.shape[2]-Trans))
    Outs_test_pred=np.zeros((RC_reps, Outs_test.shape[0], N_O, Outs_test.shape[2]-Trans))

    MSEs_train=np.zeros((RC_reps, Outs.shape[0], N_O)); 
    MSEs_test_pred=np.zeros((RC_reps, Outs_test.shape[0], N_O))

    ###also collect W_outs, Winps for preditions
    W_outs_all = []; Winps_all = []

    for r in range(RC_reps):
        ###0. Generate Winps for each RC rep
        Winps = Generate_Winps(N_I, G.number_of_nodes(), InpsNodes)

        ###1. Inital RC Training
        GNet, Init, N, W_outs = RC_Train(G, Spectral_radius, alpha, g_scale, N_I, N_O, Winps, OutsNodes, Inps, Outs, Trans, beta)
        W_outs_all.append(W_outs); Winps_all.append(Winps)

        ###2. Closed loop testing and predictions
        Outs_predict[r], MSEs_train[r] = Test_or_Predict(GNet, Init, Inps, Winps, N, N_I, N_O, OutsNodes, alpha, g_scale, W_outs, Outs, Trans)
        Outs_test_pred[r], MSEs_test_pred[r] = Test_or_Predict(GNet, Init, Inps_test, Winps, N, N_I, N_O, OutsNodes, alpha, g_scale, W_outs, Outs_test, Trans)
        
    ##### Mean over RC and standard deviation over RC reps and batches for train and test predictions
    MSEs_MnSD_train = np.array([np.mean(MSEs_train, axis=(0,1)), np.std(MSEs_train, axis=(0,1))])
    MSEs_MnSD_test_pred = np.array([np.mean(MSEs_test_pred, axis=(0,1)), np.std(MSEs_test_pred, axis=(0,1))])
    
    if Return==0:
        return MSEs_MnSD_train, MSEs_MnSD_test_pred
    
    if Return==1:
        return Outs_predict, Outs_test_pred, MSEs_MnSD_train, MSEs_MnSD_test_pred
    
    if Return==2:
        return Outs_predict, Outs_test_pred, MSEs_MnSD_train, MSEs_MnSD_test_pred, W_outs_all, Winps_all


###Training#########################################################################################   
def Ridge_Regression(R, beta, V_train, N_O, OutsNodes):
    ###R shape is (N, batch, timesteps) and V_train shape is (batch, N_O, timesteps)
    R = R.reshape(R.shape[0], -1) # (N, batch*timesteps)
    V_train = V_train.transpose(1, 0, 2).reshape(V_train.shape[1], -1) # (N_O, batch*timesteps)

    W_outs = [[]]*N_O 
    for i in range(N_O):
        R_out = R[OutsNodes[i],:]
        W_outs[i] = np.dot(np.dot(V_train[i], R_out.T), np.linalg.inv((np.dot(R_out, R_out.T) + beta*np.identity(R_out.shape[0]))))
    return W_outs

def MSE(A, B):
    return np.mean(((A - B)**2))


def Errors(y_predicted, y_actual):
    MSE = np.mean(np.square(np.subtract(y_predicted,y_actual)))
    Variance = (np.mean(np.square(np.subtract(y_actual, np.mean(y_actual) ))) )
    NMSE = MSE/Variance
    NRMSE = np.sqrt(NMSE)
    return NMSE, NRMSE 

### Testing#########################################################################################
def Test_or_Predict(GNet, Init, Inps_t, Winps, N, N_I, N_O, OutsNodes, alpha, g_scale, W_outs, Outs_test, Trans):
    
    Nodes_res = GNet.shape[0]; 
    batch, Npts_U = Inps_t.shape[0], Inps_t.shape[2]

    R = np.zeros([N, batch, Npts_U])
    Outs_pred = np.zeros((batch, N_O, Npts_U))

    for b in range(batch):
        R[:,b,0] = Init[:]
        ###Initial predicted outputs
        for i in range(N_O):
            Outs_pred[b, i, 0] = np.dot(W_outs[i], R[OutsNodes[i],b,0])

        ####time loop
        for t in range(0, Npts_U-1):                        
            Inp_term =  (Winps[0]*Inps_t[b,0,t])  
            R[:,b,t+1] = (1-alpha)*np.asarray(R[:,b,t]) + alpha*f(g_scale, np.dot(GNet, R[:,b,t]) + Inp_term)  
            ####predicting next step output based on current reservoir state and output weights
            ###also collecting the predicted outputs in Outs_pred for error calculation and plotting
            for i in range(N_O):
                Outs_pred[b, i, t+1] = np.dot(W_outs[i], R[OutsNodes[i],b,t+1])
                
    NMSEs_t = np.zeros((batch, N_O))
    for j in range(batch):
        for i in range(N_O):
            NMSEs_t[j, i] = Errors(Outs_pred[j, i, Trans:], Outs_test[j, i, Trans:])[0]
    return Outs_pred[:,:,Trans:], NMSEs_t
            