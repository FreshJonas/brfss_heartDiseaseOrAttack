import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score
from imblearn.under_sampling import RandomUnderSampler


#############################################################################################################################
# SETTINGS
#############################################################################################################################

### PATH TO DATA DIRECTORY ###
datapath = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

### HYPER PARAMETERS ###
ACTFUNC = ['logistic', 'relu']
HIDDENLAYERS = [1, 2, 3]
HIDDENLAYERSNODES = [10, 20, 50]    
SOLVER = ['adam', 'sgd']

### CHOOSE BETWEEN DROPPED NAN AND IMPUTED NAN DATASET HERE ###
# dfpath = os.path.join(datapath, '2015_cleaned_droppedNaN.csv')
# resultspath = os.path.join(datapath, '2015_results_droppedNaN.csv')
dfpath = os.path.join(datapath, '2015_cleaned_imputedNaN.csv')
resultspath = os.path.join(datapath, '2015_results_imputedNaN.csv')



#############################################################################################################################
# IMPLEMENTATION
#############################################################################################################################

# read csv
df = pd.read_csv(dfpath)

# Prepare X and y
X = df.drop('HeartDiseaseorAttack', axis=1)
y = df.HeartDiseaseorAttack

# Performing under sampling since dataset is heavily unbalanced towards people with no heart disease or attack
rus = RandomUnderSampler(random_state=1)
X_resampled, y_resampled = rus.fit_resample(X, y)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=1, test_size=0.2)

result_df = pd.DataFrame(columns=['ActFunc', 'HiddenArch', 'Solver', 'bal_acc'])


# Iterating through hyper parameters and calculating balanced accuracy score
for hyp_actfunc in ACTFUNC:

    for hyp_hiddenlayers in HIDDENLAYERS:

        for hyp_hiddenlayersnodes in HIDDENLAYERSNODES:
        
            # create hidden layer architecture tupel
            hidden_layers_architecture_tupel = (hyp_hiddenlayersnodes,)
            for i in range(hyp_hiddenlayers - 1):
                hidden_layers_architecture_tupel =  hidden_layers_architecture_tupel + (hyp_hiddenlayersnodes,)

            for hyp_solver in SOLVER:
                    
                # Create MLP Classifier
                classifier = MLPClassifier(hidden_layer_sizes=hidden_layers_architecture_tupel, solver=hyp_solver, max_iter=200, activation=hyp_actfunc, random_state=1)

                # Fitting MLP Classifier to data
                classifier.fit(X_train, y_train)

                # Predicting test data
                y_pred = classifier.predict(X_test)

                # Calculating balanced accuracy
                bal_acc = balanced_accuracy_score(y_test, y_pred)

                # Appending results to result dataframe
                result_df.loc[len(result_df.index)] = [hyp_actfunc, hidden_layers_architecture_tupel, hyp_solver, bal_acc]

                # Print current run with result
                valueDict = {'ActFunc': hyp_actfunc, 'HiddenArch': hidden_layers_architecture_tupel, 'Solver': hyp_solver, 'bal_acc': bal_acc}
                print(valueDict)

result_df.to_csv(resultspath)