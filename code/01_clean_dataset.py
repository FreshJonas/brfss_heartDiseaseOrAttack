"""
### Data Cleaning ###
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

df_original = pd.read_csv(r'../data/2015.csv')

print(df_original.head())

print(f'Database shape: {df_original.shape}')

"""# Feature Selection

Because our database is very large, we have reduced the number of features to obtain more understandable results when 
using simpler algorithms like the KNN. To do this, a brief research on the internet was done to identify the risk 
factors according to the opinion of the experts. The main sources were the United Kingdom National Health Service 
(NHS - [link](https://www.nhs.uk/conditions/cardiovascular-disease/)), the Center for Disease Control and Prevention 
(CDC - [link](https://www.cdc.gov/heartdisease/risk_factors.htm)) and the World Health Organization 
(WHO - [link](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))).

According to our researched, the following risks were selected:
1. **blood pressure (high)** code as _RFHYPE5
2. **cholesterol (high)** code as _TOLDHI2
3. **overweigh**t code as BMI5
4. **smoking** code as SMOKE100
5. **diabetes** code as DIABETE3
6. **fruit consumption** as _FRTLT1
7. **vegetables consuption** code as _VEGLT1
8. **alcohol consumption** code as _RFDRHV5 
9. **inactivity** code as _TOTINDA 
10. **age** code as _AGEG5YR
11. **gender** code as SEX
12. **targer heart disease** code as _MICHD
"""

# select specific columns
df = df_original[[
        '_MICHD', 
        '_RFHYPE5',  
        'TOLDHI2',
        '_BMI5', 
        'SMOKE100', 
        'DIABETE3', 
        '_TOTINDA', 
        '_FRTLT1', 
        '_VEGLT1', 
        '_RFDRHV5', 
        'SEX', 
        '_AGEG5YR' ]]

"""**Rename columns to make more readable**"""

# rename columns
df = df.rename(columns = {
                '_MICHD':'HeartDiseaseorAttack', 
                '_RFHYPE5':'HighBP',  
                'TOLDHI2':'HighChol', 
                '_BMI5':'BMI', 
                'SMOKE100':'Smoker', 
                'DIABETE3':'Diabetes', 
                '_TOTINDA':'PhysActivity', 
                '_FRTLT1':'Fruits', 
                '_VEGLT1':"Veggies", 
                '_RFDRHV5':'HvyAlcoholConsump', 
                'SEX':'SexIsMale', 
                '_AGEG5YR':'AgeGroup'})

############################################## ATENTION ################################################################
"""We recommend to reduce the database for testing purposes. Please uncomment the code below to reduce the dataset, 
otherwise the time required to run the code increases considerably or the computer might crushed. We recommend to run 
the complete dataset in google-CoLab since it provides more computer power. """
#df = df.loc[0:2500]
#print(f"Reduced dataset: {df.shape}")
########################################################################################################################
                
print(df.head())

"""# Handling individual features

In this section we will analyze each feature's labels following the information in the codebook. Since we still want 
to create a dataset with imputed data, we will modify both datasets df and df_droppedNaN

## Target: HeartDisease (_MICHD)

Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI). 

**Labels:**
* 1: Reported have CHD or MI.
* 2: Did not report having CHD or MI.

For logic purposes, label 2 will be substitute by 0.
"""

# HeartDiseaseorAttack (target)
df['HeartDiseaseorAttack'] = df['HeartDiseaseorAttack'].replace({2:0})

"""## Feature: HighBP (_RFHYPE5)

Adults who have been told they have high blood pressure by a doctor, nurse, or other health professional. 

**Labels:**
* 1: No
* 2: Yes
* 9: Don't know

For logic purposes, label 1 will be substitute by 0 and label 2 by 1.
"""

# HighBP
df['HighBP'] = df['HighBP'].replace({9:np.nan})
df['HighBP'] = df['HighBP'].replace({1:0})
df['HighBP'] = df['HighBP'].replace({2:1})

"""## Feature: HighChol (TOLDHI2)

Adults who have been told by a doctor, nurse or other health professional that their blood cholesterol is high

**Labels:**
* 1: Yes
* 2: No
* 7: Don't know
* 9: Refused

For logic purposes, label 1 will be substitute by 0 and label 2 by 1.
"""

# HighChol
df['HighChol'] = df['HighChol'].replace({7:np.nan})
df['HighChol'] = df['HighChol'].replace({9:np.nan})
df['HighChol'] = df['HighChol'].replace({2:0})

"""## Feature BMI

Body Mass Index (BMI).
"""

# nothing to be done here

"""## Feature Smoker (SMOKE100)

People that have smoked at least 100 cigarettes in their entire life.  

**Labels:**
* 1: yes
* 2: No
* 7: Don't know
* 9: refuse to answer

Label 2 will be substitute by 0 and label 7 and 9 will be deleted.
"""

# Smoker
df['Smoker'] = df['Smoker'].replace({7:np.nan})
df['Smoker'] = df['Smoker'].replace({9:np.nan})
df['Smoker'] = df['Smoker'].replace({2:0})

"""## Feature Diabetes (DIABETE3)

People that have been told to suffer from diabetes.  

**Labels:**
* 1: yes
* 2: yes but only during pregnancy
* 3: no
* 4: pre-diabetes or borderline diabetes. 
* 7: Don't know
* 9: refuse to answer

We will just consider yes(1) or no (3) answers. To simplify, the rest of the options will be deleted.
"""

# Diabetes
df['Diabetes'] = df['Diabetes'].replace({2:np.nan})
df['Diabetes'] = df['Diabetes'].replace({4:np.nan})
df['Diabetes'] = df['Diabetes'].replace({7:np.nan})
df['Diabetes'] = df['Diabetes'].replace({9:np.nan})
df['Diabetes'] = df['Diabetes'].replace({3:0})

"""## Feature: Physical Activity (_TOTINDA)

Adults who reported doing physical activity or exercise during the past 30days other than their regular job. 

**Labels:**
* 1: yes
* 2: no
* 9: don't know or refuse to answer. 

Label 2 will be substitute by 0 and label 9 will be deleted.
"""

# PhysActivity
df['PhysActivity'] = df['PhysActivity'].replace({2:0})
df['PhysActivity'] = df['PhysActivity'].replace({9:np.nan})

"""## Feature: Fruit consumption (_FRTLT1)

Consume Fruit 1 or more times per day. 


**Labels:**
* 1: yes
* 2: no
* 9: don't know or refuse to answer. 

Label 2 will be substitute by 0 and label 9 will be deleted.
"""

# Fruits
df['Fruits'] = df['Fruits'].replace({2:0})
df['Fruits'] = df['Fruits'].replace({9:np.nan})

"""## Feature: Veggies (_VEGLT1)

People consume vegetables 1 or more times per day. 


**Labels:**
* 1: yes
* 2: no
* 9: don't know or refuse to answer. 

Label 2 will be substitute by 0 and label 9 will be deleted.
"""

# Veggies
df['Veggies'] = df['Veggies'].replace({2:0})
df['Veggies'] = df['Veggies'].replace({9:np.nan})

"""## Feature: Alcohol consumption (_RFDRHV5)

Adult men having more than 14 drinks per week and adult women having more than 7 drinks per week. 

**Labels:**
* 1: no
* 2: yes
* 9: don't know or refuse to answer. 

Label 1 will be substitute by 0, label 2 by 1, and label 9 will be deleted.
"""

# HvyAlcoholConsump
df['HvyAlcoholConsump'] = df['HvyAlcoholConsump'].replace({1:0})
df['HvyAlcoholConsump'] = df['HvyAlcoholConsump'].replace({2:1})
df['HvyAlcoholConsump'] = df['HvyAlcoholConsump'].replace({9:np.nan})

"""## Feature: gender (sex)

Indicate the gender of respondent. 

**Labels:**
* 1: male
* 2: female

Label 2 will be substitute by 0.
"""

# SexIsMale
df['SexIsMale'] = df['SexIsMale'].replace({2:0})

"""## Feature: age group (_AGEG5YR)

There are 14- level age categories 

**Labels:**
* 1: 18 to 24 years
* 2: 25 to 29 years
* 3: 30 to 34 years
* 4: 35 to 39 years
* 5: 40 to 44 years
* 6: 45 to 49 years
* 7: 50 to 54 years
* 8: 55 to 59 years
* 9: 60 to 64 years
* 10: 65 to 69 years
* 11: 70 to 74 years
* 12: 75 to 79 years
* 13: 80 or older
* 14: don't know or refuse answering

Label 14 will be deleted.
"""

# AgeGroup
df['AgeGroup'] = df['AgeGroup'].replace({14:np.nan})

"""# Handling missing values

First we check how much percentage of the data is missing.
"""

print(round((((df.isnull().sum()).sum() / np.product(df.shape)) * 100), 2))

"""We check then, which columns have the highest percentage of missing values. """

print(df.isnull().sum())

"""Missing values can be handle with one of the following options:

1. Invest more in data collection: This option is not possible for us due to time-limitations and out of scope 
responsabilities.
2. Data exclusion: Remove rows with missing data.
3. Data imputation: Replace values by artificial new data.
4. Just leave the missing values: Not recommended.

According to the slides from our class (Block05-06 page 25), we should never impute the target variable in supervised 
learning. Therefore, we have decided to delete all rows with a missing target value (HeartDisease). 
"""

# We dropped all rows whose value for HeartDisease is Nan. Or better explain, we keep all the rows that have a
# value that is not NaN
df = df[df['HeartDiseaseorAttack'].notna()]
print(df.isnull().sum())

print(f'shape: {df.shape}')

"""## Scale Data

A critical point for most ml-algorithms as well as knn imputing is scaling the data. For simplicity, we will use 
Scikit-Learn’s MinMaxScaler which will scale our variables to have values between 0 and 1.
"""

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

df.head()

df_scaled.head()

"""## V1 Dataset with dropped NaN"""

df_droppedNaN = df_scaled.dropna()
print(df_droppedNaN.isnull().sum())

print(df_droppedNaN.shape)
print(df_scaled.shape)

# save as new csv
df_droppedNaN.to_csv('../data/2015_cleaned_droppedNaN.csv', index=False)

"""## V2 Dataset with imputed NaN

In this section we created a dataset with imputed NaN for learning purposes. We would also like to compare the 
performance between both datasets. We are aware that imputation might introduce biases.

Now we can create the KNN-Imputater
"""

imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean', add_indicator=False)

# !!!THIS TAKES VERY LONG!!! (at least one hour)
df_imputedNaN = imputer.fit_transform(df_scaled)

print(round((((df.isnull().sum()).sum() / np.product(df.shape)) * 100), 2))

print(df_imputedNaN.shape)
print(df_scaled.shape)

df_imputed = pd.DataFrame(df_imputedNaN, columns=df_scaled.columns)
df_imputed.to_csv('../data/2015_cleaned_imputedNaN.csv')

print("\nEnd of file 01_clean_dataset")