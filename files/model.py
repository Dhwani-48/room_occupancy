# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
import pickle

# loading the dataset of room occupancy
occupancy_dataset= pd.read_csv('C:\\Users\\JAYVIR CHHASATIYA\\Desktop\\Data science\\case-study-june-2022 (1)\\data\\occupancy.csv')

# seperating the data and labels
X = occupancy_dataset.drop(['Occupancy','date','HumidityRatio'],axis=1)
Y = occupancy_dataset['Occupancy']

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)
#stratify for splitting dataset which will not led to move all 1 output in X and 0 in Y it will be equal

# model using random forest classifier algorithm
pipeline_rf=Pipeline([('scalar3',StandardScaler()),
('pca3',PCA(0.95)),
('rf_classifier',RandomForestClassifier())])

pipeline_rf.fit(X_train, Y_train)

# export
pickle.dump(pipeline_rf,open('pipe.pkl','wb'))
model = pickle.load(open('pipe.pkl','rb'))


