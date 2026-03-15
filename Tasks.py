"""
Created on Feb  27.02.2026

@author: Dr. Manish Yadav
Email: manish.yadav@tu-berlin.de

This code contains the functions for the article "Emergent E-I Structure in Performance-Evolved Reservoir Networks of
Neuronal Population Dynamics" by Manish Yadav. The main functions are:
- load_WC_Neuron_Data: Loads the Wilson-Cowan neuron model data for training and testing.
- InpGenerate: Generates the input and output data for the specified task type.
- InpPlot: Plots the input and output time series for a specified number of samples.
- InPlot_with_batches: Plots the input and output time series for all batches, with transparency to visualize the variability across batches.
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from RC_Funcs import *
from PDNE_Functions import *


def load_WC_Neuron_Data(WCDataDirectory, Type='Train'):
    Tr_Inps = np.load(os.path.join(WCDataDirectory, "WC_Train_Inputs.npy"))
    Tr_Sols = np.load(os.path.join(WCDataDirectory, "WC_Train_Sols.npy"))
    
    Tst_Inps = np.load(os.path.join(WCDataDirectory, "WC_Test_Inputs.npy"))
    Tst_Sols = np.load(os.path.join(WCDataDirectory, "WC_Test_Sols.npy"))

    Prd_Inps = np.load(os.path.join(WCDataDirectory, "WC_Predict_Inputs.npy"))
    Prd_Sols = np.load(os.path.join(WCDataDirectory, "WC_Predict_Sols.npy"))
    ###inps shape (s) [5, 200] (batches, time) and sols (x, y) shape [5, 2, 200] (batches, outputs, time)
    ### Train_Inps shape [5, 1, 200] (batches, inps, time) inps are [s]
    ### Train_Outs shape [5, 2, 200] (batches, outputs, time)
    Train_Inps = np.zeros((Tr_Inps.shape[0], 1, Tr_Inps.shape[1]))
    Train_Outs = np.zeros((Tr_Sols.shape[0], 2, Tr_Sols.shape[2]))
    for i in range(Tr_Inps.shape[0]):
        Train_Inps[i, 0, :] = Tr_Inps[i, :]
        Train_Outs[i, :, :] = Tr_Sols[i, :, :]

    Test_Inps = np.zeros((Tst_Inps.shape[0], 1, Tst_Inps.shape[1]))
    Test_Outs = np.zeros((Tst_Sols.shape[0], 2, Tst_Sols.shape[2]))
    for i in range(Test_Inps.shape[0]): 
        Test_Inps[i, 0, :] = Tst_Inps[i, :]
        Test_Outs[i, :, :] = Tst_Sols[i, :, :]

    Prd_Inps_reshaped = np.zeros((Prd_Inps.shape[0], 1, Prd_Inps.shape[1]))
    Prd_Sols_reshaped = np.zeros((Prd_Sols.shape[0], 2, Prd_Sols.shape[2]))
    for i in range(Prd_Inps.shape[0]): 
        Prd_Inps_reshaped[i, 0, :] = Prd_Inps[i, :]
        Prd_Sols_reshaped[i, :, :] = Prd_Sols[i, :, :]
        
    if Type=='Train':
        return Train_Inps[:,:,:], Train_Outs[:,:,:], Test_Inps[:,:,:], Test_Outs[:,:,:]
    if Type=='Predict':
        return Prd_Inps_reshaped, Prd_Sols_reshaped
#########################################################################################

def InpGenerate(TaskType, N_I, Npts_U, InpProps):    
    if TaskType=="WC_Neuron" and InpProps[1]=='Train':
        Inps, Outs, Inps_test, Outs_test = load_WC_Neuron_Data(InpProps[0], InpProps[1])
        return Inps, Outs, Inps_test, Outs_test
    if TaskType=="WC_Neuron" and InpProps[1]=='Predict':
        Inps, Outs = load_WC_Neuron_Data(InpProps[0], InpProps[1])
        return Inps, Outs

def InpPlot(Inps, Outs, N_I):
    for i in range(N_I):  
        fig_size = plt.rcParams["figure.figsize"]  
        fig_size[0] = 8; fig_size[1] = 1.5
        plt.rcParams["figure.figsize"] = fig_size 
        plt.plot(Inps[i, :400], c='C0')
        plt.ylabel('Input', fontsize=12)
        plt.xlabel('time', fontsize=12)
        plt.show()
        plt.plot(Outs[i, :400], c='C1')
        plt.ylabel('Output', fontsize=12)
        plt.xlabel('time', fontsize=12)
        plt.show()

def InPlot_with_batches(Inps, Outs):
    for i in range(Inps.shape[1]):  
        fig_size = plt.rcParams["figure.figsize"]  
        fig_size[0] = 8; fig_size[1] = 1.5
        plt.rcParams["figure.figsize"] = fig_size 
        for j in range(Inps.shape[0]):
            plt.plot(Inps[j, i, :400], c='C0', alpha=0.5)
        plt.ylabel('Input', fontsize=12)
        plt.xlabel('time', fontsize=12)
        plt.show()
        
    for i in range(Outs.shape[1]):  
        for j in range(Outs.shape[0]):
            plt.plot(Outs[j, i, :400], c='C1', alpha=0.5)
        plt.ylabel('Output', fontsize=12)
        plt.xlabel('time', fontsize=12)
        plt.show()