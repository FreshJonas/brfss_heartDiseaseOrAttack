
"""03_check_target_balancing.ipynb

# Check Balancing of Target Variable
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path_cleaned_droppedNaN = r'../data/2015_cleaned_droppedNaN.csv'

df = pd.read_csv(path_cleaned_droppedNaN)
print(df.head())

df_target_value_counts = df.HeartDiseaseorAttack.value_counts().reset_index()
sns.barplot(x="index", y="HeartDiseaseorAttack", data=df_target_value_counts)
plt.show()
print("\nPeople with heart disease (1.0) vs. people without heart disease (0.0):")
print(df_target_value_counts)

"""## Result

Heavily unbalanced towards patients with no heart disease or attack.
This has to be taken into account for later when performing different machine learning algorithms.
"""

print("\nEnd of file 03_check_target_balancing")