# PDNE-WilsonCowanNeuron

## 1. Brief Description

**Emergent E-I Structure in Performance-Evolved Reservoir Networks of Neuronal Population Dynamics**  
**Manish Yadav**  
Chair of Cyber-Physical Systems in Mechanical Engineering, Technische Universitat Berlin, Strasse des 17. Juni 135, 10623 Berlin, Germany  
Electronic mail: `manish.yadav@tu-berlin.de`  
Dated: 13 March 2026

Understanding how network structure gives rise to neuronal dynamics, and whether compact computational models can recover that structure from data alone, is a central challenge in computational neuroscience. This repository applies the performance-dependent network evolution (PDNE) framework to the Wilson-Cowan (WC) neuronal system, a canonical two-population model of excitatory-inhibitory (E-I) interaction underlying physiological rhythms.

Starting from a minimal random seed network, PDNE iteratively grows and prunes a reservoir computing (RC) network based only on prediction performance, yielding compact and task-optimized reservoir networks. The evolved networks accurately predict both excitatory `E(t)` and inhibitory `I(t)` population activities across unseen stimulus amplitudes and generalize in a zero-shot manner to novel stimulus configurations (pulse number, position, and amplitude) without retraining.

Structural analysis in the provided notebooks shows a consistent functional organization with nodes specialized for E, I, and shared E-I representations. At population level, evolved connectivity recovers the correct WC excitatory-inhibitory sign pattern for most interaction types without being hard-coded into the architecture. Overall, the project demonstrates that performance-driven evolution can produce models that are both accurate and structurally interpretable for physiological rhythms.

## 2. Repository Organization (Files and Roles)

This repository is flat (single top-level directory, no nested source folders). Main files:

- `PDNE_Functions.py`: Core PDNE logic (network growth/pruning, checkpoint evolution loop, model run orchestration, and save/load helpers).
- `RC_Funcs.py`: Reservoir-computing engine (reservoir dynamics, spectral-radius scaling, ridge regression readout training, test/prediction, NMSE calculation).
- `Tasks.py`: Data loading and input/output generation utilities for WC datasets (`Train`, `Test`, `Predict` modes).
- `Plots.py`: Visualization utilities for trajectories, network graph structure, network measures, and performance over evolution.
- `PDNE_RunTasks.ipynb`: Main execution notebook to set parameters and run multi-repetition PDNE evolution.
- `PDNE_Load_And_Predic.ipynb`: Postprocessing notebook to load saved runs/models, evaluate final networks, and analyze predictive/generalization behavior.
- `WC_Neuron_Simul.ipynb`: Wilson-Cowan simulation notebook used to generate/inspect response dynamics and bifurcation-style analyses.
- `WC_Neuron.ode`: ODE definition of the Wilson-Cowan E-I system (including parameters and continuation settings).
- `LICENSE`: Repository license.
- `README.md`: Project overview and usage notes.

Data note:

- The workflow expects pre-generated WC data files such as `WC_Train_Inputs.npy`, `WC_Train_Sols.npy`, `WC_Test_Inputs.npy`, `WC_Test_Sols.npy`, `WC_Predict_Inputs.npy`, and `WC_Predict_Sols.npy` in a data directory referenced by notebook parameters (for example `DataDir`).

## 3. Main Functionality

The project implements an end-to-end PDNE + RC pipeline:

1. Generate or load an initial random reservoir seed network (`RandNetTestGenerator` in `RC_Funcs.py`).
2. Initialize input/readout node assignments (`InpOut_Init_Gen`).
3. Load WC train/test/predict datasets (`InpGenerate` and `load_WC_Neuron_Data` in `Tasks.py`).
4. Iteratively evolve network topology with PDNE (`Checkpoint_V3` in `PDNE_Functions.py`):
	- Add candidate nodes/links (`AddNewNode`) if prediction error improves.
	- Delete nodes (`DeleteNode`) when pruning does not degrade performance.
5. Train and evaluate RC readouts with ridge regression (`RC`, `RC_Train`, `Ridge_Regression` in `RC_Funcs.py`).
6. Track and visualize model quality and structure (`Plot_Performance`, `Plot_NetMsrs`, `Net_Plot`, trajectory plots).
7. Save full evolution history and final models for later analysis (`SaveData`, `save_final_model`).

Primary execution entrypoint used in notebooks:

- `Run_Full_Model(...)` in `PDNE_Functions.py`.

## 4. Required Python Libraries

Core required packages:

- `numpy`
- `scipy`
- `matplotlib`
- `networkx`
- `jupyter` (for running `.ipynb` workflows)

Notebook-only or optional packages:

- `tqdm` (used in `WC_Neuron_Simul.ipynb`)

Standard-library modules used (no install needed):

- `os`, `time`, `timeit`, `copy`, `pickle`, `math`

Minimal install command:

```bash
pip install numpy scipy matplotlib networkx jupyter tqdm
```

