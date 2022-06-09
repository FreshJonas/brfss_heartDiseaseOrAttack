import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
"""**Upload Data**

We upload the data in a pandas dataframe.
"""

# Path to dataset with dropped NaN values
local_dropped_path = r'../../data/2015_cleaned_droppedNaN.csv'
# Path to dataset with imputed NaN values
local_imputed_path = '../../pics/2015_cleaned_imputedNaN.csv'
df = pd.read_csv(local_dropped_path)
"""# Pre-Processing Data"""

print(df.head())

print(df.shape)

"""**Check empty values**"""

print(round((((df.isnull().sum()).sum() / np.product(df.shape)) * 100), 2))
"""**We recommend to reduce the database for testing purposes. Please uncomment the code below to reduce the dataset,
otherwise the time required to run the code increases considerably or the computer might crushed. We recommend to run
the complete dataset in google-CoLab since it provides more computer power.**"""
df = df.loc[0:2500]
print(df.shape)

"""**Split Data**"""

X = df.drop(columns=['HeartDiseaseorAttack'])
y = np.array(df['HeartDiseaseorAttack'])

"""**Balance Data**

We performed undersampling because our target is not balanced. 
"""

rus = RandomUnderSampler(random_state=1)
X_resampled, y_resampled = rus.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.2)

print(y_train.shape)

print(X_train.shape)

"""# KNN

We have previously tested the performance with StandardScaler() and MinMaxScaler() and we have obtained, 
that the performance is very similar or identical. Since the data has been already scaled with MinMaxScaler() for 
the KNN-Imputer, we will not scaled the data again here.
"""


"""

### optimize KNN with GridSearchCV

We tried to optimized the KNN using RandomizedSearchCV. We will try to get the best number of neighbors.
"""


knn = KNeighborsClassifier()

"""We generate a list of possible values for k, which are all odd to avoid possible "ties"."""

k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

params = {'n_neighbors': k, 'metric': ['manhattan', 'euclidean', 'minkowski']}

"""cv=5 means cross validation with 5 folds"""

random_search = GridSearchCV(knn, params, cv=5)
random_search.fit(X_train, y_train)

"""Call score_samples on the estimator with the best found parameters."""

print("The best parameters are:", random_search.best_params_)

best_k = random_search.best_params_.get('n_neighbors')
print(best_k)

metric = random_search.best_params_.get('metric')

"""Check if we get the same or very similar result: """

accuracy_list =[]
for i in range(100):
  X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)
  knn = KNeighborsClassifier(n_neighbors=best_k, metric = metric)
  knn.fit(X_train, y_train)
  prediction = knn.predict(X_test)

  balanced_accuracy_score(y_test, prediction)
  accuracy_list.append(balanced_accuracy_score(y_test, prediction))
print("Balanced Accuracy Score is: ", np.mean(accuracy_list))

"""# Other Evaluations

## Confusion Matrix
"""

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(knn, X_test, y_test)
plt.show()

collector = []
for k in range(1, 20, 2):
  accuracy_list = []
  for j in range(1, 10): # I run 10 times and take the mean accuracy
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    accuracy = balanced_accuracy_score(y_test, prediction)
    accuracy_list.append(accuracy)
  collector.append({"k": k,
                    "accuracy": round(np.mean(accuracy_list), 3).astype('float64')})

accuracy_scores_df = pd.DataFrame(collector)

plt.plot(accuracy_scores_df['k'], accuracy_scores_df['accuracy'])
plt.grid(True)
plt.show()
print("\nThe End")