*** project in progress
# A Neural Network-Aided Functional Model of Photovoltaic Arrays for a Wide Range of Atmospheric Conditions

![Parameters dependency over a wide operating range for PV module](https://raw.githubusercontent.com/DIE-UTFSM-AA/A-Neural-Network-Aided-Functional-Model-of-PVArrays-for-a-Wide-Range-of-Atmospheric-Conditions/refs/heads/main/FIgs/fig2.png)


## Abstrac
As the cost of photovoltaic (PV) power generation declines and becomes competitive in the electricity business, there is an increasing need to accurately predict the performance of this technology under a wide range of operating conditions. The performance of a PV module may be captured via its current-voltage ($\textit{I}-\textit{V}$) characteristic. The single-diode model is an adequate  approximation of this characteristic when the parameters are determined for the atmospheric conditions at which the curve was measured. However, capturing the dependency of these parameters so that the model can reproduce $\textit{I}-\textit{V}$ characteristics for a wide range of atmospheric conditions is a challenging task. The objective of this paper is to develop such model. To accomplish this task, a large-scale data repository consisting of climatic and operational measurements is used to train an artificial neural network (NN) that captures the behavior of each parameter. The trained NN is then utilized to recreate $\textit{I}-\textit{V}$ curves for a broad spectrum of environmental conditions. The analysis of the parameter behavior and the curves predicted by the NN model allows the identification of an improved PV model by searching through a kernel of functions. As the results show, the proposed model outperforms current functional models available in the literature, by reducing the error in power estimation by about 6\% when measured for a wide operating range.


## Installation
For the installation and start-up of the repository, it is necessary to have the following libraries:

* [tensorflow](https://www.tensorflow.org/install/pip)
* [tensorflow-probability](https://www.tensorflow.org/probability/install)
* [scikit-learn](https://scikit-learn.org/stable/install.html)
* [numpy](https://numpy.org/install/)
* [pandas](https://pandas.pydata.org/docs/getting_started/install.html)
* [sympy](https://www.sympy.org/en/index.html)






### Database:
The data used for the development of this work are obtained in [Data for Validating Models for PV Module Performance](https://datahub.duramat.org/dataset/data-for-validating-models-for-pv-module-performance) of NREL






## Contents


![Reconstruction of the behavior of the reciprocal of the modified nonideality factor over a wide range of operating conditions for PV module](https://raw.githubusercontent.com/DIE-UTFSM-AA/A-Neural-Network-Aided-Functional-Model-of-PVArrays-for-a-Wide-Range-of-Atmospheric-Conditions/refs/heads/main/FIgs/fig1.png)






## Citation
    @ARTICLE{10187624,
      author={Angulo, Alejandro and Huerta, Miguel and Mancilla–David, Fernando},
      journal={IEEE Transactions on Industrial Informatics}, 
      title={A Neural Network–Aided Functional Model of Photovoltaic Arrays for a Wide Range of Atmospheric Conditions}, 
      year={2023},
      volume={},
      number={},
      pages={1-9},
      abstract={As the cost of photovoltaic (PV) power generation declines and becomes competitive in the electricity business, there is an increasing need to accurately predict the performance of this technology under a wide range of operating conditions. The performance of a PV module may be captured via its current–voltage (I–V) characteristic. The single–diode model is an adequate approximation of this characteristic when the parameters are determined for the atmospheric conditions at which the curve was measured. However, capturing the dependency of these parameters so that the model can reproduce I–V characteristics for a wide range of atmospheric conditions is a challenging task. The objective of this paper is to develop such model. To accomplish this task, a large-scale data repository consisting of climatic and operational measurements is used to train an artificial neural network (NN) that captures the behavior of each parameter. The trained NN is then utilized to recreate I–V curves for a broad spectrum of environmental conditions. The analysis of the parameter behavior and the curves predicted by the NN model allows the identification of an improved PV model by searching through a kernel of functions. As the results show, the proposed model outperforms current functional models available in the literature, by reducing the error in power estimation by about 6% when measured for a wide operating range.},
      keywords={},
      doi={10.1109/TII.2023.3285048},
      ISSN={1941-0050},
      month={},}

    
