# A Neural Network-Aided Functional Model of Photovoltaic Arrays for a Wide Range of Atmospheric Conditions

![Parameters dependency over a wide operating range for PV module](https://github.com/DIE-UTFSM-AA/A-Neural-Network-Aided-Functional-Model-of-PVArrays-for-a-Wide-Range-of-Atmospheric-Conditions/blob/main/FIgs/fig2.png)


## Abstrac
As the cost of photovoltaic (PV) power generation declines and becomes competitive in the electricity business, there is an increasing need to accurately predict the performance of this technology under a wide range of operating conditions. The performance of a PV module may be captured via its current-voltage ($\textit{I}-\textit{V}$) characteristic. The single-diode model is an adequate  approximation of this characteristic when the parameters are determined for the atmospheric conditions at which the curve was measured. However, capturing the dependency of these parameters so that the model can reproduce $\textit{I}-\textit{V}$ characteristics for a wide range of atmospheric conditions is a challenging task. The objective of this paper is to develop such model. To accomplish this task, a large-scale data repository consisting of climatic and operational measurements is used to train an artificial neural network (NN) that captures the behavior of each parameter. The trained NN is then utilized to recreate $\textit{I}-\textit{V}$ curves for a broad spectrum of environmental conditions. The analysis of the parameter behavior and the curves predicted by the NN model allows the identification of an improved PV model by searching through a kernel of functions. As the results show, the proposed model outperforms current functional models available in the literature, by reducing the error in power estimation by about 6\% when measured for a wide operating range.


## Getting Started
An easy way to get started ...

* [GEM cookbook](https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master//examples/environment_features/GEM_cookbook.ipynb)
* [Keras-rl2 example](https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master/examples/reinforcement_learning_controllers/keras_rl2_dqn_disc_pmsm_example.ipynb)
* [Stable-baselines3 example](https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master/examples/reinforcement_learning_controllers/stable_baselines3_dqn_disc_pmsm_example.ipynb)
* [MPC  example](https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master/examples/model_predictive_controllers/pmsm_mpc_dq_current_control.ipynb)


## Contents

![Reconstruction of the behavior of the reciprocal of the modified nonideality factor over a wide range of operating conditions for PV module](https://github.com/DIE-UTFSM-AA/A-Neural-Network-Aided-Functional-Model-of-PVArrays-for-a-Wide-Range-of-Atmospheric-Conditions/blob/main/FIgs/fig1.png)






## Database:
[Data for Validating Models for PV Module Performance](https://datahub.duramat.org/dataset/data-for-validating-models-for-pv-module-performance)






## Citation
    cite{}
    
