# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:20:56 2023

@author: Dr. Manish Yadav
Email: manish.yadav@tu-berlin.de

This code contains the functions for the article "Emergent E-I Structure in Performance-Evolved Reservoir Networks of
Neuronal Population Dynamics" by Manish Yadav. The main functions are:
1. Network_Measures(G): to calculate the network measures of a given graph G
2. Node deletion function: to delete a random node and check if the performance improves or not, if yes then update the network and properties and scores
3. Node addition function: to add a new node with random connections and check if the performance improves or not, if yes then update the network and properties and scores
4. Checkpoint main function: to run the main loop of the model, where in each iteration, nodes can be added or deleted based on performance improvements
5. SaveData function: to save the data of the model evolution for analysis and plotting
"""

import pickle
from RC_Funcs import *
from Tasks import *
from PDNE_Functions import *
from Plots import *

    
##### Network Measures and performances  #########################################################################################
def Network_Measures(G):
    Msr_names = ['Nodes', 'Avg. CC', 'Avg. In_Deg', 'Avg. Out_Deg', 'Communities', 'Density']
        
    Nodes = G.number_of_nodes()
    CC = nx.average_clustering(G)
    DegIn = np.mean(np.array(list(dict(G.in_degree()).values())))
    DegOut = np.mean(np.array(list(dict(G.out_degree()).values())))
    Communities = len(nx.community.greedy_modularity_communities(G))
    Density = nx.density(G)
    return np.array([Nodes, CC, DegIn, DegOut, Communities, Density]), Msr_names
       
#########################################################################################
#########################################################################################
   
def DeleteNode(G_old, alpha, g_scale_old, N_I, N_O, InpsNodes_old, OutsNodes_old, Spectral_radius, Inps, Outs, Inps_test, Outs_test, \
               Trans, RC_Reps, NetMsr_init, MSEs_Mn_pred_old, Err_precision, NodesDel_Percent, beta):
    
    Max_DeleteSteps = round((NodesDel_Percent*G_old.number_of_nodes())/100)
    
    NetMeasrs=np.array([]); Scores_train=[]; Scores_pred=[]
    DeleteStep=0; Flag=0; Nodes_Deleted=0
    while(Flag==0):
        
        ### Make copies of G_old, InpsNodes_old, N_I and N_O ##################
        G_temp = G_old.copy()
        InpsNodes_temp = copy.deepcopy(InpsNodes_old)
        OutsNodes_temp = copy.deepcopy(OutsNodes_old)
        g_scale_temp = copy.deepcopy(g_scale_old)
        ### Delete random node#######################################
        G_Nodes = np.array(G_temp.nodes)
        Del_node = np.random.choice(G_Nodes)
        G_temp.remove_node(Del_node)
        
        #### Rename/shift the indexing of nodes and edges###########
        mapping = dict(zip(G_temp, range(0, G_temp.number_of_nodes())))
        G_temp = nx.relabel_nodes(G_temp, mapping)
        
        #### If deleted node was also an input node#################
        for ni in range(N_I):
            # print('Before shifting:', InpsNodes_temp, 'ni:', ni, InpsNodes_temp[ni])
            if Del_node in InpsNodes_temp[ni]:
                InpsNodes_temp[ni].remove(Del_node)
                # print('Deleted node:', Del_node, 'was also inp node. Inp nodes:', InpsNodes_temp)           
            
            ##### shifting inp nodes index number########################
            for i in range(len(InpsNodes_temp[ni])):
                if(InpsNodes_temp[ni][i]>=Del_node):
                    InpsNodes_temp[ni][i] = InpsNodes_temp[ni][i]-1 

        ###remove the g_scale of the deleted node as well
        g_scale_temp = list(np.delete(g_scale_temp, Del_node))

        # print('After shifting:', InpsNodes_temp)
        ################################################################
        
        #### If deleted node was also an Output node#################
        for no in range(N_O):
            # print('O-Before shifting:', OutsNodes_temp, 'no:', no, OutsNodes_temp[no])
            if Del_node in OutsNodes_temp[no]:
                OutsNodes_temp[no].remove(Del_node)
                # print('O-Deleted node:', Del_node, 'was also out node. Out nodes:', OutsNodes_temp)           
            
            ##### shifting inp nodes index number########################
            for i in range(len(OutsNodes_temp[no])):
                if(OutsNodes_temp[no][i]>=Del_node):
                    OutsNodes_temp[no][i] = OutsNodes_temp[no][i]-1 
        ################################################################
        
        ### RC run####################################################################
        MSEs_MnSD_train_temp, MSEs_MnSD_pred_temp = RC(G_temp, Spectral_radius, alpha, g_scale_temp, N_I, N_O, InpsNodes_temp, OutsNodes_temp,\
                                                        Inps, Outs, Inps_test, Outs_test, Trans, RC_Reps, beta)
        
        ######## Rounding the errors for comparing till given nth digit
        ######## If improvement in error in any of the output then update the Network, Inpnodelist and errors 
        ######## but other errors should not deplete
        Temp_Err_Precise=np.round(MSEs_MnSD_pred_temp[0], Err_precision)
        Old_Err_Precise=np.round(MSEs_Mn_pred_old, Err_precision)
        # print('Del fun condn:')
        # print('MSEs_MnSD_pred_temp[0]:',MSEs_MnSD_pred_temp[0],Temp_Err_Precise,'MSEs_Mn_pred_old:',MSEs_Mn_pred_old,Old_Err_Precise)
        if (Temp_Err_Precise <= Old_Err_Precise).all(): 
            G_old = G_temp.copy()    
            Nodes_Deleted = Nodes_Deleted+1
            InpsNodes_old = InpsNodes_temp
            OutsNodes_old = OutsNodes_temp
            g_scale_old = g_scale_temp
            MSEs_Mn_pred_old = MSEs_MnSD_pred_temp[0]
            
            ######## Net. properties and scores########
            NetMeasrs, NetMsrs_Names = Network_Measures(G_old)
            Scores_train=MSEs_MnSD_train_temp
            Scores_pred=MSEs_MnSD_pred_temp
            # print('Deletion Accepted.', 'Nodes:', G_old.number_of_nodes(), 'Links:', G_old.number_of_edges(), 'Inp Nodes:', InpsNodes_old\
            #       ,'Out Nodes:', OutsNodes_old, 'OldErr:', MSEs_Mn_pred_old,'TempErr:', MSEs_MnSD_pred_temp[0])
                        
        DeleteStep=DeleteStep+1
        if DeleteStep>=Max_DeleteSteps:
            Flag=1
                    
    return G_old, g_scale_old, InpsNodes_old, OutsNodes_old, NetMeasrs, Scores_train, Scores_pred, Nodes_Deleted

# @jit(target_backend='cuda')   
def AddNewNode(t, G_old, alpha, g_scale_old, N_I, N_O, MaxNewLinks, Psi, P_inp, P_out, InpsNodes_old, OutsNodes_old, InpNodeType, OutNodeType, Spectral_radius,\
               Inps, Outs, Inps_test, Outs_test, Trans, RC_Reps, MSEs_Mn_pred_old, Err_precision, Max_AddSteps, beta):
    MaxNewLinks_arr = np.arange(1, MaxNewLinks+1)
    Flag=0; AddStep=0
    while(Flag==0):
        
        ##### Making copies of G_old Input nodes list
        G_temp = G_old.copy()
        InpsNodes_temp = copy.deepcopy(InpsNodes_old)
        OutsNodes_temp = copy.deepcopy(OutsNodes_old)
        g_scale_temp = copy.deepcopy(g_scale_old)
        
        ##### Add new node
        NewNode = G_temp.number_of_nodes()
        G_temp.add_node(NewNode)
        G_Nodes = np.array(G_temp.nodes())
        #### New node can be input node for each input with Prob P_inp 'separately' #################
        N = G_temp.number_of_nodes()

        #### Add activation function scale g_scale for new node, drawn from given list
        g_scale_new = np.random.uniform(0.01, 1.0)
        g_scale_temp.append(g_scale_new)
        
        if (InpNodeType==0):
            ##### New node can be input node for other inputs as well (in multiple tasks) with Prob P_inp
            for i in range(N_I):
                if np.random.random()<=P_inp:
                    InpsNodes_temp[i].append(N-1)
        
        if (InpNodeType==1):
            ##### New node can be input node for any 'one' input (in multiple tasks) with Prob P_inp, Strict Exclusiveness
            InpNums=np.arange(N_I)
            Inp_rand=np.random.random()
            if Inp_rand<=P_inp:
                Which_InpNum_Gets_InpNode=np.random.choice(InpNums)
                InpsNodes_temp[Which_InpNum_Gets_InpNode].append(N-1)
            ######################################################################
        
        if (OutNodeType==0):
            #### New node can be output node for each output with Prob P_out 'separately' (in multiple tasks)#################
            for i in range(N_O):
                if np.random.random()<=P_out:
                    OutsNodes_temp[i].append(N-1)
       
        if (OutNodeType==1):
            ##### New node can be output node for any 'one' output (in multiple tasks) with Prob P_out, Strict Exclusiveness
            OutNums=np.arange(N_O)
            Out_rand=np.random.random()
            if Out_rand<=P_out:
                Which_OutNum_Gets_OutNode=np.random.choice(OutNums)
                OutsNodes_temp[Which_OutNum_Gets_OutNode].append(N-1)
        ######################################################################
           
        NewConnsNum = np.random.choice(MaxNewLinks_arr)
        for n in range(NewConnsNum):
            ###draw weights from given list    
            Drawn_Weight = np.random.uniform(-1.0, 1.0)  #(-0.1, 1.0)
            if(np.random.rand()<=Psi):
                G_temp.add_edge(NewNode, np.random.choice(G_Nodes), weight=Drawn_Weight)
            else:
                G_temp.add_edge(np.random.choice(G_Nodes), NewNode, weight=Drawn_Weight)

        ### RC run
        MSEs_MnSD_train_temp, MSEs_MnSD_pred_temp = RC(G_temp, Spectral_radius, alpha, g_scale_temp, N_I, N_O, InpsNodes_temp, OutsNodes_temp, \
                                                       Inps, Outs, Inps_test, Outs_test, Trans, RC_Reps, beta)
        
        ######## Rounding the errors for comparing till given nth digit
        ######## If improvement in error in any of the output then update the Network, Inpnodelist and errors 
        ######## but other errors should not deplete
        Temp_Err_Precise=np.round(MSEs_MnSD_pred_temp[0], Err_precision)
        Old_Err_Precise=np.round(MSEs_Mn_pred_old, Err_precision)
        
        ##### IMPORTANT: '<=' condition will be required for multiple tasks, as all should not improve at the same time. 
        ##### Some improve, some can stay exactly same but should not degrade 
        if (Temp_Err_Precise < Old_Err_Precise).all():  
            t=t+1
            G_old = G_temp.copy()
            InpsNodes_old = InpsNodes_temp
            OutsNodes_old = OutsNodes_temp
            g_scale_old = g_scale_temp
            # print(t, 'Nodes Added.', 'Nodes:', G_temp.number_of_nodes(), 'Links:', G_temp.number_of_edges(), 'Inp Nodes:', InpsNodes_old,\
            #       'Out nodes:',OutsNodes_old, 'Old Err:', Old_Err_Precise, 'Temp Err:', Temp_Err_Precise)
            Flag=1
        else:
            if AddStep>=Max_AddSteps:
                t=t+1
                Flag=1
            AddStep=AddStep+1
            ############################################################
            # print(t, 'Addition Rejected.', 'Nodes:', G_old.number_of_nodes(), 'Links:', G_old.number_of_edges(), 'Inp Nodes:', InpsNodes_old,\
            #       'Out nodes:',OutsNodes_old, 'Old Err:',Old_Err_Precise, 'Temp Err:', Temp_Err_Precise)
    
    return G_old, g_scale_old, InpsNodes_old, OutsNodes_old, t

def remove_isolated_nodes(G, InpsNodes, OutsNodes, N_I, N_O):
    IsolateFlag=0
    if (nx.number_of_nodes(G) > 3) and (list(nx.isolates(G))):
        Isolated_Nodes = list(nx.isolates(G))
        G.remove_nodes_from(Isolated_Nodes)
        ##### If any of the isolated nodes are also input or output nodes, remove them from the respective lists as well
        for ni in range(N_I):
            for node in Isolated_Nodes:
                if node in InpsNodes[ni]:
                    InpsNodes[ni].remove(node)
        for no in range(N_O):
            for node in Isolated_Nodes:
                if node in OutsNodes[no]:
                    OutsNodes[no].remove(node)
        IsolateFlag=1
        print('Isolated nodes:', Isolated_Nodes, 'also removed.')
    return G, InpsNodes, OutsNodes, IsolateFlag

# @jit(target_backend='cuda')   
def Checkpoint_V3(Net_Init, alpha, g_scale, MaxNewLinks, Psi, P_inp, P_out, N_I, N_O, InpsNodes, OutsNodes, Spectral_radius, T, Delta_Err, beta, PlotEvery, Inps, \
                  Outs, Inps_test, Outs_test, Trans, RC_Reps, Err_precision, Max_AddSteps, NodesDel_Percent, Informed_Growth='Yes',\
                  Delete_Nodes='Yes', InpNodeType=0, OutNodeType=0):
    
    ###preserve the weights of the initial network 
    G = nx.DiGraph(Net_Init)
            
    ### Init RC run
    MSEs_MnSD_train, MSEs_MnSD_pred = RC(G, Spectral_radius, alpha, g_scale, N_I, N_O, InpsNodes, OutsNodes, Inps, Outs, Inps_test, \
                                         Outs_test, Trans, RC_Reps, beta)
    print(0, 'Initial Net. Nodes:', G.number_of_nodes(), 'Links:', G.number_of_edges(), 'Err:', MSEs_MnSD_pred[0])
    ################ Initial Net., properties and performance################################################
    ###Lists to keep track of the evolution of the network and its properties and performance
    AllGraphs=[]
    AllGraphs.append(G)
    All_g_scales = []
    All_g_scales.append(g_scale)
    AllInpsNodes=[]; AllOutsNodes=[]
    NetMsr_init, NetMsrs_Names = Network_Measures(G)
    NetMsrs = np.zeros((T+10, NetMsr_init.shape[0]))
    NetMsrs[0] = NetMsr_init
    ##########################################################################################################

    ##### MSEs_MnSD_train.Shape[0]->2(mean and std), MSEs_MnSD_train.Shape[1]->N_O
    ##### Scores_train.shape[0]->time, shape[1]->2(mean and std), shape[2]->N_O
    Scores_train = np.zeros((T+10, MSEs_MnSD_train.shape[0], MSEs_MnSD_train.shape[1]))+np.nan
    Scores_pred = np.zeros((T+10, MSEs_MnSD_pred.shape[0], MSEs_MnSD_pred.shape[1]))+np.nan
    Scores_train[0] = MSEs_MnSD_train
    Scores_pred[0] = MSEs_MnSD_pred
    
    # print(0, 'Nodes:', G.number_of_nodes(), 'Links:', G.number_of_edges(), ' Err:', NetMsrs[0, 8])
    #########################################################################################################
    ######While loop until desired error is reached or taking too long Time
    t=0
    while( (Scores_pred[t,0,:] > Delta_Err).any() ):

        ###### Delete Node#####################################################################################################
        if (Delete_Nodes=='Yes'):
            
            G, g_scale, InpsNodes, OutsNodes, NetMsrs_AfterDelFun, Scores_train_AfterDelFun, Scores_pred_AfterDelFun, Nodes_deleted =\
            DeleteNode(G, alpha, g_scale, N_I, N_O, InpsNodes, OutsNodes, Spectral_radius, Inps, Outs, Inps_test, Outs_test,\
            Trans, RC_Reps, NetMsr_init, Scores_pred[t,0], Err_precision, NodesDel_Percent, beta)
            
            if Nodes_deleted>=1:
                t=t+1
                AllGraphs.append(G)
                All_g_scales.append(g_scale)
                AllInpsNodes.append(InpsNodes); AllOutsNodes.append(OutsNodes)
                NetMsrs[t] = NetMsrs_AfterDelFun.T
                Scores_train[t] = Scores_train_AfterDelFun
                Scores_pred[t] = Scores_pred_AfterDelFun

            print(t, 'After Delete 1 Fun : Updated Net. Nodes:', G.number_of_nodes(), 'Links:', G.number_of_edges(),\
                  'Inp Nodes:',[len(InpsNodes[i]) for i in range(N_I)],'Out nodes:',[len(OutsNodes[i]) for i in range(N_O)],\
                    'Deleted Nodes:', Nodes_deleted,'g_scale:', len(g_scale),'Err:',Scores_pred[t,0])
            ##### Skip if t already reach ==T #######################
            if t>=T:  break
        ##########################################################################################################################
        
        #####Add new node#########################################
        if (Informed_Growth=='Yes'):   
            
            G, g_scale, InpsNodes, OutsNodes, t = AddNewNode(t, G, alpha, g_scale, N_I, N_O, MaxNewLinks, Psi, P_inp, P_out, InpsNodes, OutsNodes,\
            InpNodeType, OutNodeType, Spectral_radius, Inps, Outs, Inps_test, Outs_test, Trans, RC_Reps, Scores_pred[t,0],\
            Err_precision, Max_AddSteps, beta)  # Err_precision BEFORE Max_AddSteps

        ####### Net. properties
        MSEs_MnSD_train, MSEs_MnSD_pred = RC(G, Spectral_radius, alpha, g_scale, N_I, N_O, InpsNodes, OutsNodes, Inps, Outs, Inps_test, Outs_test,\
                                             Trans, RC_Reps, beta)
        
        AllGraphs.append(G)
        All_g_scales.append(g_scale)
        AllInpsNodes.append(InpsNodes); AllOutsNodes.append(OutsNodes)
        Scores_train[t] = MSEs_MnSD_train
        Scores_pred[t] = MSEs_MnSD_pred
        NetMsrs[t], NetMsrs_Names = Network_Measures(G)
        
        print(t, 'After Addition fun: Updated Net. Nodes:', G.number_of_nodes(), 'Links:', G.number_of_edges(),\
              'Inp Nodes:', [len(InpsNodes[i]) for i in range(N_I)],'Out nodes:',[len(OutsNodes[i]) for i in range(N_O)], 'g_scale:', len(g_scale),'Err:', Scores_pred[t,0])
        #########################################################
        
        ###### Delete Node Only during last step if condition is met####################################################################
        if ((Scores_pred[t,0,:] < Delta_Err).any() and Delete_Nodes=='Yes'):
        
            G, g_scale, InpsNodes, OutsNodes, NetMsrs_AfterDelFun, Scores_train_AfterDelFun, Scores_pred_AfterDelFun, Nodes_deleted =\
            DeleteNode(G, alpha, g_scale, N_I, N_O, InpsNodes, OutsNodes, Spectral_radius, Inps, Outs, Inps_test, Outs_test,\
            Trans, RC_Reps, NetMsr_init, Scores_pred[t,0], Err_precision, NodesDel_Percent, beta)

            #### check for isolated nodes and remove them as well, as they will not contribute to the performance and just add to the network size and complexity.
            IsolateFlag=0
            
            if Nodes_deleted>=1:
                t=t+1
                ###recalculate the network measures and scores
                NetMsrs_AfterDelFun, NetMsrs_Names = Network_Measures(G)
                Scores_train_AfterDelFun, Scores_pred_AfterDelFun = RC(G, Spectral_radius, alpha, g_scale, N_I, N_O, InpsNodes, OutsNodes, Inps, \
                                                                       Outs, Inps_test, Outs_test, Trans, RC_Reps, beta) 
                ####update the network and properties and scores

                AllGraphs.append(G)
                All_g_scales.append(g_scale)
                AllInpsNodes.append(InpsNodes); AllOutsNodes.append(OutsNodes)
                NetMsrs[t] = NetMsrs_AfterDelFun.T
                Scores_train[t] = Scores_train_AfterDelFun
                Scores_pred[t] = Scores_pred_AfterDelFun

            print(t, 'After Delete 2 Fun : Updated Net. Nodes:', G.number_of_nodes(), 'Links:', G.number_of_edges(),\
                  'Inp Nodes:',[len(InpsNodes[i]) for i in range(N_I)],'Out nodes:',[len(OutsNodes[i]) for i in range(N_O)],\
                    'Deleted Nodes:', Nodes_deleted,'g_scale:', len(g_scale),'Err:',Scores_pred[t,0])
            
            ##### Skip if t already reach ==T #######################
            if t>=T:  break
        ##########################################################################################################################
        
        #### Plot
        if (t%PlotEvery==0):
            Net_Plot(G,5)
        ##### Skip if t already reach ==T #######################
        if t>=T:  break
        ########################################################
    return AllGraphs, All_g_scales, NetMsrs, NetMsrs_Names, Scores_train, Scores_pred, AllInpsNodes, AllOutsNodes

def write_gpickle_compat(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def SaveData(SaveDir, TaskType, NetProps, InpProps, ModelRep, NetMsr, NetMsr_Names, Scores, AllGraphs, All_g_scales, AllInpsNodes, AllOutsNodes):
    
    np.save(os.path.join(SaveDir,'NetMeasures_Names.npy'), NetMsr_Names)
    if TaskType=='WC_Neuron':
        np.save(os.path.join(SaveDir,'{:}_NetMeasures_Rp{:d}.npy'.format(TaskType, ModelRep)), NetMsr) 
        np.save(os.path.join(SaveDir,'{:}_Scores_Rp{:d}.npy'.format(TaskType, ModelRep)), Scores) 

        write_gpickle_compat(AllInpsNodes, os.path.join(SaveDir,'{:}_InpsNodes_Rp{:d}.gpickle'.format(TaskType, ModelRep)))
        write_gpickle_compat(AllOutsNodes, os.path.join(SaveDir,'{:}_OutsNodes_Rp{:d}.gpickle'.format(TaskType, ModelRep)))
        write_gpickle_compat(AllGraphs, os.path.join(SaveDir,'{:}_Graphs_Rp{:d}.gpickle'.format(TaskType, ModelRep)))
        write_gpickle_compat(All_g_scales, os.path.join(SaveDir,'{:}_g_scales_Rp{:d}.gpickle'.format(TaskType, ModelRep)))


def save_final_model(G, g_scale, InpsNodes, OutsNodes, W_outs, Winps, Predictions, SaveDir, TaskType, ModelRep, Name='Final'):
    write_gpickle_compat(G, os.path.join(SaveDir,'{:}_{:}_Graph_Rp{:d}.gpickle'.format(TaskType, Name, ModelRep)))
    write_gpickle_compat(g_scale, os.path.join(SaveDir,'{:}_{:}_g_scale_Rp{:d}.gpickle'.format(TaskType, Name, ModelRep)))
    write_gpickle_compat(InpsNodes, os.path.join(SaveDir,'{:}_{:}_InpsNodes_Rp{:d}.gpickle'.format(TaskType, Name, ModelRep)))
    write_gpickle_compat(OutsNodes, os.path.join(SaveDir,'{:}_{:}_OutsNodes_Rp{:d}.gpickle'.format(TaskType, Name, ModelRep)))
    write_gpickle_compat(W_outs, os.path.join(SaveDir,'{:}_{:}_W_outs_Rp{:d}.gpickle'.format(TaskType, Name, ModelRep)))
    write_gpickle_compat(Winps, os.path.join(SaveDir,'{:}_{:}_Winps_Rp{:d}.gpickle'.format(TaskType, Name, ModelRep)))
    write_gpickle_compat(Predictions, os.path.join(SaveDir,'{:}_{:}_Predictions_Rp{:d}.gpickle'.format(TaskType, Name, ModelRep)))

def read_gpickle_compat(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
##################################################################################################################################################################################
##### Run full model#########################################################################################

def ModelType(Informed_Growth, Delete_Nodes):
    if Informed_Growth=='No' and Delete_Nodes == 'No':
        ModelTyp='N1'
    if Informed_Growth=='Yes' and Delete_Nodes == 'No':
        ModelTyp='N2'
    if Informed_Growth=='Yes' and Delete_Nodes == 'Yes':
        ModelTyp='N3'
    return ModelTyp
    
# @jit(target_backend='cuda')   
def Run_Full_Model(Net_Init, alpha, g_scale_init, Npts_U, TaskType, NetProps, InpProps, MaxNewLinks, Psi, P_inp,\
    P_out, N_I, N_O, InpsNodes_init, OutsNodes_init, Spectral_radius, T, Delta_err, T_plot,\
    Transs, RC_Reps, Err_precision, Max_AddSteps, NodesDel_Percent, Informed_Growth, Delete_Nodes,\
    InpNodeType, OutNodeType, Model_Reps, Scores_Names, SaveDir, SaveDataFlag, beta, DataDir):
    
    ModelTyp = ModelType(Informed_Growth, Delete_Nodes)
    
    for Mr in range(Model_Reps):

        ###Step 1: load initial network from DataDir and generate inputs and outputs for the task
        Net_Init = read_gpickle_compat(os.path.join(DataDir,'Net_Init_{:d}.gpickle'.format(Mr)))
        
        ###convert Net_Init to DiGraph if it is not already
        if not isinstance(Net_Init, nx.DiGraph):
            Gnet_init = nx.DiGraph(Net_Init)
        else:
            Gnet_init = Net_Init

        InpsNodes_init, OutsNodes_init = InpOut_Init_Gen(Net_Init, P_inp, P_out, N_I, N_O)
        print('Generated Initial Input nodes:', InpsNodes_init)
        print('Generated Initial Output or Readout nodes:', OutsNodes_init)
        ###plot initial network
        print('Initial Network:')
        Net_Plot(Gnet_init,7)

        # start = timer()
        print('Model rep: ', Mr)
        Inps, Outs, Inps_test, Outs_test = Task.InpGenerate(TaskType, N_I, Npts_U, InpProps)
        ###if Inps has 3 dims
        if Inps is not None and len(Inps.shape)==3:
            Task.InPlot_with_batches(Inps, Outs)  
        else:
            Task.InpPlot(Inps, Outs, N_I)

        AllGraphs, All_g_scales, NetMsrs, NetMsrs_Names, Scores_train, Scores_pred, AllInpsNodes, AllOutsNodes = Checkpoint_V3(Net_Init, alpha, g_scale_init, MaxNewLinks, Psi, P_inp,\
        P_out, N_I, N_O, InpsNodes_init, OutsNodes_init, Spectral_radius, T, Delta_err, beta, T_plot, Inps, Outs, Inps_test,Outs_test,Transs,\
        RC_Reps, Err_precision, Max_AddSteps, NodesDel_Percent, Informed_Growth, Delete_Nodes, InpNodeType, OutNodeType)
        Scores = np.array([Scores_train, Scores_pred])
        print('Completed Model Rep:', Mr)#, "completed, time elapsed:", (timer()-start)/60, 'mins')
        #### Save Data#######################################
        if(SaveDataFlag=="Yes"):
            SaveData(SaveDir, TaskType, NetProps, InpProps, Mr, NetMsrs, NetMsrs_Names, Scores, AllGraphs, All_g_scales, AllInpsNodes, AllOutsNodes)
        
        #### Print resuts and plots#################################################
        print('\n################################################################')
        print('\nNetwork Evolution Model Type:',ModelTyp)
        print('\nRepetition',Mr+1,'of',TaskType,'task is completed!!!')
        print('\nThe final network contains ',AllGraphs[-1].number_of_nodes(),'nodes and',\
              AllGraphs[-1].number_of_edges(),'edges.')
        print('Final Input nodes:', AllInpsNodes[-1])
        print('Final Output or Readout nodes:', AllOutsNodes[-1])
        Plot_NetMsrs(NetMsrs, NetMsrs_Names)
        Plot_Performance(Scores, Scores_Names, N_O)
        
        Tillt=1000
        print('Final Evolved Network:')
        ###pring final net matrix 
        Net_Plot(AllGraphs[-1],7)


        ##### Output from final Net.####################################
        print('\nClose-loop prediction final network:')
        Outs_predict, Outs_test_pred, MSEs_train, MSEs_pred, W_outs_all, Winps_all = RC(AllGraphs[-1], Spectral_radius, alpha, All_g_scales[-1], N_I, N_O, AllInpsNodes[-1], AllOutsNodes[-1], Inps, Outs,\
                                                               Inps_test, Outs_test, Transs, 1, beta=beta, Return=2)
        
        ###Save final best model separately             
        if(SaveDataFlag=="Yes"):
            save_final_model(AllGraphs[-1], All_g_scales[-1], AllInpsNodes[-1], AllOutsNodes[-1], W_outs_all, Winps_all, Outs_test_pred, SaveDir, TaskType, Mr)
            
        ##print Pred shape and MSEs
        print('Predicted test output shape:', Outs_predict.shape, 'Test Predicted output shape:', Outs_test_pred.shape)
        print('MSEs for train outputs:', MSEs_train)
        
        Plot_t(N_O, Outs, Outs_predict[0], Transs, Tillt, MSEs_train[0], TaskType, 'Training')
        Plot_t(N_O, Outs_test, Outs_test_pred[0], Transs, Tillt, MSEs_pred[0], TaskType, 'Predictions')


        ####Final Predictions####################################################################################
        InpProps_pred = [DataDir, 'Predict']
        Inps_predict, Outs_predict = Task.InpGenerate(TaskType, N_I, Npts_U, InpProps_pred)

        print('\nClose-loop prediction final network:')
        Outs_predict_f, Outs_test_pred_f, MSEs_train_f, MSEs_pred_f, W_outs_all_f, Winps_all_f = \
            RC(AllGraphs[-1], Spectral_radius, alpha, All_g_scales[-1], N_I, N_O, AllInpsNodes[-1], AllOutsNodes[-1], Inps, Outs,\
                                                               Inps_predict, Outs_predict, Transs, 1, beta=beta, Return=2)
        
        ###Save final best model separately             
        if(SaveDataFlag=="Yes"):
            save_final_model(AllGraphs[-1], All_g_scales[-1], AllInpsNodes[-1], AllOutsNodes[-1], W_outs_all_f, Winps_all_f, Outs_test_pred_f, SaveDir, TaskType, Mr, Name='Final_DiversePredictions')
            
        ##print Pred shape and MSEs
        print('Final Predicted output shape:', Outs_predict_f.shape, 'Final Test Predicted output shape:', Outs_test_pred_f.shape)
        print('Final MSEs for train outputs:', MSEs_train_f)
        
        Plot_t(N_O, Outs, Outs_predict_f[0], Transs, Tillt, MSEs_train_f[0], TaskType, 'Training')
        Plot_t(N_O, Outs_predict, Outs_test_pred_f[0], Transs, Tillt, MSEs_pred_f[0], TaskType, 'All Final Predictions')


    ###return best final models for prediction and analysis
    return AllGraphs[-1], All_g_scales[-1], AllInpsNodes[-1], AllOutsNodes[-1], W_outs_all_f, Winps_all_f

