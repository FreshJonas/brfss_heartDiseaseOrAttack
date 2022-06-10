# brfss_heartDiseaseOrAttack

Link to Dataset
https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system?select=2015.csv

Please run the files in the following order: 
1. 01_clean_dataset.py
2. 02_feature_influence.py
3. 03_check_target_balancing.py

This 3 files are designed to preprocess the data and generate the datasets that will be used by the models.
The code for each model is in the corresponding folder. 

## Installation
You might need to install the following libraries: 
* six
* pydotplus: it provides a Python Interface to Graphviz's Dot language, which we use to create graphs (decision-tree). 
* graphviz. You might have to include the Graphviz executables on your system's PATH
