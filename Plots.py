"""
Created on Feb  27.02.2026

@author: Dr. Manish Yadav
Email: manish.yadav@tu-berlin.de

This code contains the functions for the article "Emergent E-I Structure in Performance-Evolved Reservoir Networks of
Neuronal Population Dynamics" by Manish Yadav. The main functions are:
1. Plot_t: This function plots the original and predicted trajectories of the system for different tasks (Chaos, VDP, WC_Neuron). It also plots the error between the original and predicted trajectories.
2. Net_Plot: This function plots the network structure of the reservoir using NetworkX. It visualizes the weights of the connections between neurons, with negative weights shown as positive for better visualization.
3. Plot_NetMsrs: This function plots the evolution of network measures (e.g., spectral radius, clustering coefficient) during the performance evolution process.
4. Plot_Performance: This function plots the performance (e.g., NMSE) of the reservoir during training and prediction phases for different numbers of outputs. It shows how the performance evolves over iterations.
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


def Plot_t(N_O, Outs, Outs_t, Trans, Tillt, MSE_t, TaskType, Label):
    
    if (TaskType=='Chaos') or (TaskType=='VDP'):
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 6; fig_size[1] = 6
        plt.rcParams["figure.figsize"] = fig_size
        plt.plot(Outs[0,Trans:Tillt+Trans], Outs[1,Trans:Tillt+Trans], lw=0.5, label='Original')
        plt.plot(Outs_t[0,:Tillt], Outs_t[1,:Tillt], lw=0.5, c='r', label=Label)
        plt.legend(loc='best', fontsize=16)
        plt.show()
    
        for i in range(N_O):
            fig_size = plt.rcParams["figure.figsize"]  
            fig_size[0] = 10; fig_size[1] = 2.5
            plt.rcParams["figure.figsize"] = fig_size  

            plt.plot(Outs[i,Trans:Tillt+Trans], lw=1, label='Original')
            plt.title('Error={:.10f}'.format(MSE_t[i]))
            plt.plot(Outs_t[i,:Tillt],ls='--', lw=1.75, label=Label)
            plt.ylabel('Output', fontsize=14)
            plt.xlabel('time', fontsize=14)
            plt.legend(loc='best', fontsize=16)
            plt.show()

    if TaskType=='WC_Neuron':
        fig_size = plt.rcParams["figure.figsize"]  
        fig_size[0] = 10; fig_size[1] = 2.5
        plt.rcParams["figure.figsize"] = fig_size  

        ###loop over RC reps
        for i in range(Outs.shape[0]):
            ###shapes of Outs and Outs_t are (N_O, batch, time), making new panel for each batch
            for b in range(Outs.shape[1]):
                plt.plot(Outs[i,b,Trans:Tillt+Trans], lw=1, label='Original')
                plt.plot(Outs_t[i,b,:Tillt],ls='--', lw=1.75, label=Label)
                plt.ylabel('Output', fontsize=14)
                plt.xlabel('time', fontsize=14)
                plt.legend(loc='best', fontsize=16)
                plt.show()



def Net_Plot(G,gs):  
    ###make it weights are negative, then make them positive for better visualization

    G_plot = copy.deepcopy(G)
    for u, v, d in G_plot.edges(data=True):
        if d['weight']<0:
            G_plot[u][v]['weight'] = -d['weight']
    ## plot
    fig_size = plt.rcParams["figure.figsize"]   
    fig_size[0] = 8; fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size
    ###make kamada-kawai layout for better visualization
    pos = nx.kamada_kawai_layout(G_plot)
    nx.draw_networkx_nodes(G_plot, pos, node_size=250, node_color='red', alpha=0.6)
    ###write original weights on edges not from G_plot, but from G, to show negative weights as well
    edge_labels = {(u, v): '{:.2f}'.format(d['weight']) for u, v, d in G.edges(data=True)}
    ###adjust self loop edge labels to show them outside the node
    for u, v, d in G.edges(data=True):
        if u==v:
            edge_labels[(u, v)] = '{:.2f}'.format(d['weight'])

    nx.draw_networkx_edges(G_plot, pos, width=0.6)
    # nx.draw_networkx_edge_labels(G_plot, pos, edge_labels=edge_labels, font_size=10)
    nx.draw_networkx_labels(G_plot, pos, labels=None, font_size=12)
    plt.box(False)
    plt.show()


def Plot_NetMsrs(NetMsrs, NetMsrs_Names):
    fig_size = plt.rcParams["figure.figsize"]  
    fig_size[0] = 14; fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size  
    fig, ax = plt.subplots(2, 3)
    
    Colr = ['r', 'blue', 'orange', 'green', 'brown', 'cyan', 'mediumpurple']
    LastEvol_t = np.where(NetMsrs[:,0] == 0)[0][0]-1
    k=0
    for i in range(2):
        for j in range(3):
            ax[i, j].plot(NetMsrs[:LastEvol_t,k], lw=1.5,c=Colr[k], label=NetMsrs_Names[k])
            ax[i, j].legend(loc='upper right',fontsize=14)
            ax[i,j].tick_params(axis='both',labelsize=12)
            ax[i,j].set_xlabel('Iterations', fontsize=16)
            k+=1
    plt.show()
    

def Plot_Performance(Scores, Scores_Names, N_O):
    ### Scores.shape[0]-> 2(train, pred), shape[1]->time, shape[2]->2(mean and std), shape[3]->N_O
    fig_size = plt.rcParams["figure.figsize"]  
    fig_size[0] = 10; fig_size[1] = 4
    plt.rcParams["figure.figsize"] = fig_size  
    fig, ax = plt.subplots(1, 2)
    
    Colr=['C0','C1']
    k=0
    #### loop for train, test
    for i in range(2):
        #### loop for Num of Outputs performances
        for j in range(N_O):
            ### meanplots
            LastEvol_t = np.argwhere(np.isnan(Scores[i][:,0,j]))[0][0]-1
            ax[i].plot(Scores[i][:LastEvol_t,0,j], lw=2, c=Colr[i], label='{:}Out{:d}'.format(Scores_Names[i],j+1))
            ax[i].legend(loc='upper right',fontsize=14)
            ax[i].tick_params(axis='both',labelsize=12)
            ax[i].set_ylabel('NMSE', fontsize=16)
            ax[i].set_xlabel('Iterations', fontsize=16)
    ax[0].set_ylim(-0.02, 1.05)

    ##if Save Data flag is Yes, then save the performance plot as well
    # if SaveFig:
    #     plt.savefig(os.path.join(SaveDir,'{:}_Performance_Rp{:d}.png'.format(TaskType, ModelRep)), dpi=300)

    plt.show()
    