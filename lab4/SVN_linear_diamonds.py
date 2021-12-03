## Authors: Piotr Michalek (s19333), Jan Kibort (s19916)
## Predict diamond price based on its weight
## data source https://www.kaggle.com/shivam2503/diamonds?select=diamonds.csv&fbclid=IwAR3EIZfTs1utYZM-mB1b1sYqkdVchfgDN6d_oz35YjYSjUqHpjQOYmUT0ZE
## to run from console: python ./SNV_linear_diamonds.py

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

df_raw=pd.read_csv('./diamonds.csv',sep='delimiter', header=None,  engine='python')

## Dropping intial junk values,renaming the column and resetting the index values
df = df_raw.drop([0,], axis=0).reset_index(drop=True).rename(columns={0:'Carat'})
df = df.Carat.str.split(',',expand=True).rename(columns={0:'Carat', 1:'Price'})
df.head()

## Convert the columns to Numberic values
df.Carat = pd.to_numeric(df.Carat, errors='coerce')
df.Price = pd.to_numeric(df.Price, errors='coerce')

## Segregating the No of claims to various categories
df["Carat_Category"] = pd.cut(df["Carat"],
                               bins=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0, 3.5, 4.0, np.inf],
                                labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

## Droping the NAN record
df = df.dropna().reset_index(drop=True)

## Applying the Stratifiedshufflesplit and ensuring Carat_category is 
## equally spread across training and testing set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["Carat_Category"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
    
## removing the Carat_Category column as this column is no longer needed as it was primarly created for the shuffle split
for set_ in (strat_train_set, strat_test_set):
    set_.drop("Carat_Category", axis=1, inplace=True)
    
## Assigning Training and testing set logical names
df_prepared = strat_train_set.drop("Price", axis=1)
df_labeled =  strat_train_set.drop("Carat", axis=1)

test_prepared = strat_test_set.drop("Price", axis=1)
test_labeled = strat_test_set.drop("Carat", axis=1)

## Running the Linear regression model on the test set
lin = LinearRegression().fit(test_prepared, test_labeled)

## Show graph
plt.rcParams["figure.figsize"] = (10,6)
plt.scatter(df_prepared, df_labeled)
plt.plot(df_prepared,lin.predict(df_prepared),color='red')
plt.xlabel('Carat')
plt.ylabel('Price ($)')
plt.show()