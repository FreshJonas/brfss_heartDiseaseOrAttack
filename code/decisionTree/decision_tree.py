
# To make nice plots we recommend install these libraries:
#!pip install six
#!pip install pydotplus
#!pip install graphviz


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn import tree
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn import metrics
from six import StringIO
import pydotplus
from graphviz import Digraph

"""# Pre-Processing data

Get Dataset
"""
# Paths:
collabs_dropped_path = r'/content/drive/MyDrive/Colab Notebooks/DS/2015_cleaned_droppedNaN.csv'
collabs_imputed_path = r'/content/drive/MyDrive/Colab Notebooks/DS/2015_cleaned_imputedNaN.csv'
# Note: if you use the dataset with imputed NaN values, the features BMI and AgeGroup are already scaled.
# Path to dataset with dropped NaN values
local_dropped_path = r'../../data/2015_cleaned_droppedNaN.csv'
# Path to dataset with imputed NaN values
local_imputed_path = '../../data/2015_cleaned_imputedNaN.csv'
df = pd.read_csv(local_dropped_path)

print(df.head())

print(f"Dataset shape is {df.shape}")

"""**Check empty values**"""
print("\nPercentage of empty values is:")
print(round((((df.isnull().sum()).sum() / np.product(df.shape)) * 100), 2))

############################ ATENTION ################################################
"""We recommend to reduce the database for testing purposes. Please uncomment the code below to reduce the dataset, 
otherwise the time required to run the code increases considerably or the computer might crushed. We recommend to run 
the complete dataset in google-CoLab since it provides more computer power. """
# df = df.loc[0:2500]
# print(f"Reduced dataset: {df.shape}")
########################################################################################

X = df.drop(columns=['HeartDiseaseorAttack'])
y = np.array(df['HeartDiseaseorAttack'])

"""**Balance Data**

We performed undersampling because our target is not balanced.
"""

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(X, y)

"""**Data splitting**
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.2)

print(f'shape of y: {y_train.shape}')
print(f'shape of y: {X_train.shape}')


"""# Gini vs Entropy

In this section we analyze if the method to calculate impurities (gini or entropy) affects the accuracy score.
"""

gini_scores = []
entropy_scores = []

for i in range (0, 100): 
  gini_model = DecisionTreeClassifier(criterion='gini', random_state=1)
  entropy_model = DecisionTreeClassifier(criterion='entropy', random_state=1)

  gini_model.fit(X_train, y_train)
  entropy_model.fit(X_train, y_train)

  gini_predictions = gini_model.predict(X_test)
  entropy_prediction = entropy_model.predict(X_test)

  gini_scores.append(balanced_accuracy_score(y_test, gini_predictions))
  entropy_scores.append(balanced_accuracy_score(y_test, entropy_prediction))

avg_scores_gini = sum(gini_scores) / len(gini_scores)
avg_scores_entropy = sum(entropy_scores) / len(entropy_scores)

print(f"\nAverage accuracy score for gini {avg_scores_gini}")
print(f"Average accuracy score for entropy {avg_scores_entropy}")

print("\nStatistical Analysis for gini")
gini_statistics = pd.Series(gini_scores)
print(gini_statistics.describe())

print("\nStatistical Analysis for entropy")
entropy_statistics = pd.Series(entropy_scores)
print(entropy_statistics.describe())

"""Tree plot for gini"""

gini_model = DecisionTreeClassifier(criterion='gini', random_state=1)
gini_model.fit(X_train, y_train)

tree.plot_tree(gini_model)
dot_data = StringIO()

# We extract the column names to label correctly the data in the graph:
column_names = []
for column in X: 
  column_names.append(column)
clases_names = []
for c in gini_model.classes_:
  clases_names.append(str(c))

export_graphviz(gini_model ,out_file=dot_data, filled=True, rounded=True, 
                special_characters=True, class_names=clases_names, feature_names=column_names)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("tree_gini_not_optimized.png")

"""Tree plot for entropy"""

entropy_model = DecisionTreeClassifier(criterion='entropy', random_state=1)
entropy_model.fit(X_train, y_train)
tree.plot_tree(entropy_model)
dot_data = StringIO()

# For label correctly the data in the nodes: 
clases_names = []
for c in entropy_model.classes_:
  clases_names.append(str(c))

export_graphviz(entropy_model ,out_file=dot_data, filled=True, rounded=True, 
                special_characters=True, class_names=clases_names, feature_names=column_names)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("tree_entropy_not_optimized.png")

"""**Conclusion**

Entropy seems to have a better performance for this particular dataset. However, the gini criterion is faster because 
it is less computationally expensive. With a really big dataset (like our dataset) it might not be worth the time 
invested in training when using the entropy criterion. 

Moreover both trees seem overfitted, therefore we will analyze pruning in the next section.

# PRUNING THE TREE

Pruning means limiting the growth of a tree with the purpose of avoiding overfitting. 

Decision-trees classifiers in sklearn use the following parameters for pruning: 
* max_depth
* max_leaf_nodes
* min_samples_split
* min_samples_leaf
* min_impurity_decrease

For our tree, we will directly use ccp (cost complexity pruning), which is a post-pruning technique. The subtree with 
the largest cost complexity that is smaller than ccp_alpha will be chosen. 

The higher alpha is, the more the tree is pruned. An alpha of 0 will not preformed pruning (will leave just a node 
in the tree).
"""

model = DecisionTreeClassifier(criterion='gini', random_state=1)

"""We create a list of alpha values to be tested in the tree model. """

ccp_alphas = np.arange(0, 0.5, 0.02)
print("\nAlpha list is")
print(ccp_alphas)

model_alphas = []
for alpha in ccp_alphas: 
  model = DecisionTreeClassifier(ccp_alpha=alpha, random_state=1)
  model.fit(X_train, y_train)
  model_alphas.append(model)

"""Now I graph the accuracy of each tree using the Training Dataset and the Testing Dataset as a function of alpha. 

The blue line is the accuracy for the training dataset. 
The yellow line is the accuracy for the test dataset. 

As we prune (alpha gets bigger) we see that the training accuracy decreases but the accuracy of testing increase. 
"""

train_scores = [model.score(X_train, y_train) for model in model_alphas]
test_scores = [model.score(X_test, y_test) for model in model_alphas]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title('Accuracy vs alpha for training and testing sets')
ax.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle='steps-post')
ax.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle='steps-post')
ax.legend()
plt.show()

""" From previous testing, we know that the optimal value of alpha must be around 0.025. We updated the list ccp_alphas 
and we will try to find a good value for alpha using cross validation. """

ccp_alphas = np.arange(0, 0.09, 0.002)
print("\nAlpha list is")
print(ccp_alphas)

from sklearn.model_selection import cross_val_score
stat_values = []
for alpha in ccp_alphas:
  model = DecisionTreeClassifier(ccp_alpha=alpha, random_state=1)
  scores = cross_val_score(model, X_train, y_train, cv=5)
  stat_values.append([alpha, np.mean(scores), np.std(scores)])

"""Now we draw a graph for the means and standard deviation of the accuracy scores calculated for each candidate. """

alpha_results = pd.DataFrame(stat_values, columns=['alpha', 'mean_accuracy', 'std'])
alpha_results.plot(x = 'alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')
plt.show()

print("\nAlpha Results are")
print(alpha_results)

"""We could repeat the operation for a value of alpha between 0.001 and 0.0025. But since the accuracy does not seem 
to increase that much, we have decided assign 0.002 to alpha. """

max_index = alpha_results['mean_accuracy'].idxmax()
ideal_ccp_alpha = alpha_results['alpha'][max_index]
print(f"\nThe ideal value of alpha is {ideal_ccp_alpha}")

"""# Build and evaluate classification tree"""

scores = []
model_pruned = DecisionTreeClassifier(criterion='gini', ccp_alpha=ideal_ccp_alpha, random_state=1)
for i in range(0, 100):
  model_pruned = DecisionTreeClassifier(criterion='gini', ccp_alpha=ideal_ccp_alpha, random_state=1)
  model_pruned.fit(X_train, y_train)
  predictions = model_pruned.predict(X_test)
  scores.append(balanced_accuracy_score(y_test, predictions))
average_score = sum(scores) / len(scores)
print(f'\nAverage Accuracy score for pruned tree is {average_score} and its statistic is: ')
print('difference between max and mit value is ', max(scores) - min(scores))
scores = pd.Series(scores)
print(scores.describe())

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model_pruned, X_test, y_test)
plt.show()

dot_data = StringIO()
print(column_names)
export_graphviz(model_pruned ,out_file=dot_data, filled=True, rounded=True, 
                special_characters=True, class_names=clases_names, feature_names=column_names)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("tree_optimized_pruning.png")
tree.plot_tree(model_pruned)
plt.show()

print("\nThe End")