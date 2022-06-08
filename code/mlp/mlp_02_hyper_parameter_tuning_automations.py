import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score
from imblearn.under_sampling import RandomUnderSampler

# Hyper Parameters
SCALING = ['false', 'standard', 'minmax']
ACTFUNC = ['logistic', 'relu']
MAXITERATIONS = [100, 400]
HIDDENLAYERS = [1, 2, 3]
HIDDENLAYERSNODES = [10, 20, 50]    
SOLVER = ['adam', 'sgd']

# relative path to data directory
datapath = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
dfpath = os.path.join(datapath, '2015_cleaned_droppedNaN.csv')
resultspath = os.path.join(datapath, '2015_results_droppedNaN.csv')




# read csv
df = pd.read_csv(dfpath)

# Prepare X and y
X = df.drop('HeartDiseaseorAttack', axis=1)
y = df.HeartDiseaseorAttack

# Performing under sampling since dataset is heavily unbalanced towards people with no heart disease or attack
rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(X, y)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=1, test_size=0.2)

result_df = pd.DataFrame(columns=['Scaler', 'ActFunc', 'HiddenArch', 'Solver', 'MaxIter', 'bal_acc'])

for hyp_scaler in SCALING:
    
    # SCALING
    if hyp_scaler != 'false':
        if hyp_scaler == 'standard':
            scaler = StandardScaler()
        if hyp_scaler == 'minmax':
            scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    for hyp_actfunc in ACTFUNC:

        for hyp_hiddenlayers in HIDDENLAYERS:

            for hyp_hiddenlayersnodes in HIDDENLAYERSNODES:
            
                # create hidden layer architecture tupel
                hidden_layers_architecture_tupel = (hyp_hiddenlayersnodes,)
                for i in range(hyp_hiddenlayers - 1):
                    hidden_layers_architecture_tupel =  hidden_layers_architecture_tupel + (hyp_hiddenlayersnodes,)

                for hyp_solver in SOLVER:

                    for hyp_maxiterations in MAXITERATIONS:
                
                        classifier = MLPClassifier(hidden_layer_sizes=hidden_layers_architecture_tupel, solver=hyp_solver, max_iter=hyp_maxiterations, activation=hyp_actfunc, random_state=1)
                        classifier.fit(X_train, y_train)

                        y_pred = classifier.predict(X_test)

                        bal_acc = balanced_accuracy_score(y_test, y_pred)

                        # print(f'scaler: {hyp_scaler}')
                        # print(f'activation function: {hyp_actfunc}')
                        # print(f'architecture: {hidden_layers_architecture_tupel}')
                        # print(f'solver: {hyp_solver}')
                        # print(f'max iterations: {hyp_maxiterations}')
                        # print(f'bal_acc: {bal_acc}')
                        # print()

                        valueDict = {'Scaler': hyp_scaler, 'ActFunc': hyp_actfunc, 'HiddenArch': hidden_layers_architecture_tupel, 'Solver': hyp_solver, 'MaxIter': hyp_maxiterations, 'bal_acc': bal_acc}
                        result_df = result_df.append(valueDict, ignore_index=True)

result_df.to_csv(resultspath)