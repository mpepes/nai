## Authors: Piotr Michalek (s19333), Jan Kibort (s19916)

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

df_raw=pd.read_csv('./data.csv',sep='delimiter', header=None,  engine='python')

## Dropping intial junk values,renaming the column and resetting the index values
df = df_raw.drop([0, 1, 2, 3], axis=0).reset_index(drop=True).rename(columns={0:'No_Of_Claims'})
df = df.No_Of_Claims.str.split(',',expand=True).rename(columns={0:'No_Of_Claims', 1:'Total_Payment'})
df.head()

## Convert the columns to Numberic values
df.No_Of_Claims = pd.to_numeric(df.No_Of_Claims, errors='coerce')
df.Total_Payment = pd.to_numeric(df.Total_Payment, errors='coerce')

## Segregating the No of claims to various categories
df["No_Of_Claims_Category"] = pd.cut(df["No_Of_Claims"],
                               bins=[0.0, 15, 30, 45, 60, np.inf],
                                labels=[1, 2, 3, 4, 5])

## Droping the NAN record
df = df.dropna().reset_index(drop=True)

## Applying the Stratifiedshufflesplit and ensuring No_of_claims_category is 
## equally spread across training and testing set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["No_Of_Claims_Category"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
    
## removing the No_Of_Claims_Category column as this column is no longer needed as it was primarly created for the shuffle split
for set_ in (strat_train_set, strat_test_set):
    set_.drop("No_Of_Claims_Category", axis=1, inplace=True)
    
## Assigning Training and testing set logical names
df_prepared = strat_train_set.drop("Total_Payment", axis=1)
df_labeled =  strat_train_set.drop("No_Of_Claims", axis=1)

test_prepared = strat_test_set.drop("Total_Payment", axis=1)
test_labeled = strat_test_set.drop("No_Of_Claims", axis=1)

## Running the Linear regression model on the test set
lin = LinearRegression().fit(test_prepared, test_labeled)

## Show graph
plt.rcParams["figure.figsize"] = (8,6)
plt.scatter(df_prepared, df_labeled)
plt.plot(df_prepared,lin.predict(df_prepared),color='red')
plt.xlabel('Number of claims')
plt.ylabel('Payment (K)')
plt.show()