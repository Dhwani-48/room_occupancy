# room_occupancy

# Room Occupancy Detection using Machine Learning

## Problem Statement

To identify whether there is any person present in a room or not




## Authors

- [@jayvirsinhchhasatiya](https://www.github.com/jayvirsinhchhasatiya)
- [@Dhwani-48](https://www.github.com/Dhwani-48)


## About Problem

- Identify occupancy
- Bussiness Understanding
  - Application
    - Military operation
    -	To see if there is thief in the house 
    - Save energy
- Dataset
  - Available in model folder


## Setting python environment

- We use python for creating ML model

### Importing the Dependencies

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
```

### Loading Dataset

```python
occupancy_dataset = pd.read_csv("C:\\Users\\JAYVIR CHHASATIYA\\Desktop\\Data science\\case-study-june-2022 (1)\\data\\occupancy.csv")

```
### Checking info of features
```python
occupancy_dataset.info()

```
- All feature has numerical datatype

### Checking for missing values
```python
occupancy_dataset.isna().sum()

```
- There is no missing values in our dataset
- There is no categorical data

### checking any negative values present or not
```python
occupancy_dataset[occupancy_dataset['Temperature']<0].sum()
occupancy_dataset[occupancy_dataset['Humidity']<0].sum()
occupancy_dataset[occupancy_dataset['Light']<0].sum()
occupancy_dataset[occupancy_dataset['CO2']<0].sum()
occupancy_dataset[occupancy_dataset['HumidityRatio']<0].sum()
```
### plotting the graph of all the features for data analysis

- Temperature V/S Occupancy

```python
plt.bar(occupancy_dataset['Temperature'],occupancy_dataset['Occupancy'])

```

- Humidity V/S Occupancy

```python
plt.bar(occupancy_dataset['Humidity'],occupancy_dataset['Occupancy'])

```

- Light V/S Occupancy

```python
plt.bar(occupancy_dataset['Light'],occupancy_dataset['Occupancy'])

```

- CO2 V/S Occupancy

```python
plt.bar(occupancy_dataset['CO2'],occupancy_dataset['Occupancy'])

```

- HumidityRatio V/S Occupancy

```python
plt.bar(occupancy_dataset['HumidityRatio'],occupancy_dataset['Occupancy'])

```

### heatmap of dataset to know correlation between all features
```python
sns.heatmap(occupancy_dataset.corr(), annot=True)

```
- Humidity Ratio is weakly related to room occupancy.

### calculating mean 
```python
occupancy_dataset.groupby('Occupancy').mean()

```

### getting satistical analysis of data

```python
occupancy_dataset.describe()

```

### seperating the dependent and independent features

```python
X = occupancy_dataset.drop(['Occupancy','date','HumidityRatio'],axis=1) #indepedent variables
Y = occupancy_dataset['Occupancy'] # dependent variables

```

### train test split function 

```python
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)

```

### shape of X_test and X_train

```python
print(X.shape, X_train.shape, X_test.shape)

```

### model using svm algorithm

```python
pipeline_svm=Pipeline([('scalar1',StandardScaler()),
('pca1',PCA(0.95)),
('svm_classifier',svm.SVC(kernel= 'linear'))])

```

### model using logistic regression algorithm

```python
pipeline_lr=Pipeline([('scalar2',StandardScaler()),
('pca2',PCA(0.95)),
('lr_classifier',LogisticRegression())])

```

### model using random forest classifier algorithm

```python
pipeline_rf=Pipeline([('scalar3',StandardScaler()),
('pca3',PCA(0.95)),
('rf_classifier',RandomForestClassifier())])

```

### compering all model

```python
pipelines = [pipeline_svm, pipeline_lr, pipeline_rf]

# declaring the variables
best_accuracy=0.0
best_classifier=0
best_pipeline=""

# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0: 'SVM', 1: 'Logistic Regression', 2: 'RandomForest'}

# Fit the pipelines
for pipe in pipelines:
	pipe.fit(X_train, Y_train)

# declaring prediction variables	
prediction_of_test_svm= pipeline_svm.predict(X_test)
prediction_of_test_lr= pipeline_lr.predict(X_test)
prediction_of_test_rf= pipeline_rf.predict(X_test)

# printing test accuracy
for i,model in enumerate(pipelines):
    print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(X_test,Y_test)))
print(" Test Accuracy: ",(prediction_of_test_svm))
print(" Test Accuracy: ",(prediction_of_test_lr))
print(" Test Accuracy:",(prediction_of_test_rf))
```

### selecting the best model

```python
for i,model in enumerate(pipelines):
    if model.score(X_test,Y_test)>best_accuracy:
        best_accuracy=model.score(X_test,Y_test)
        best_pipeline=model
        best_classifier=i
print('Classifier with best accuracy: {}'.format(pipe_dict[best_classifier]))
```

### Drawing confusion metrices 

- Confusion Matrix of SVM(support vector machine

```python
print("Confusion Matrix of SVM(support vector machine:")
con_mat = confusion_matrix(y_true=Y_test, y_pred=prediction_of_test_svm)

ConfusionMatrixDisplay.from_predictions(y_true=Y_test, y_pred=prediction_of_test_svm)
```

- Confusion Matrix of Logistic Regression

```python
print("Confusion Matrix of Logistic Regression")
con_mat = confusion_matrix(y_true=Y_test, y_pred=prediction_of_test_lr)

ConfusionMatrixDisplay.from_predictions(y_true=Y_test, y_pred=prediction_of_test_lr)
```
- Confusion Matrix of Random Forest Classifier

```python
print("Confusion Matrix of Random Forest Classifier")
con_mat = confusion_matrix(y_true=Y_test, y_pred=prediction_of_test_rf)

ConfusionMatrixDisplay.from_predictions(y_true=Y_test, y_pred=prediction_of_test_rf)
```

### printing classification report
- Classification report of SVM
```python
svm_report = classification_report(Y_test,prediction_of_test_svm)
print(svm_report)
```
- Classification report of Logistic Regression

```python
lr_report = classification_report(Y_test,prediction_of_test_lr)
print(lr_report)
```
- Classification report of Random Forest Classifier

```python
rf_report = classification_report(Y_test,prediction_of_test_rf)
print(rf_report)
```

### Exporting model
```python
pickle.dump(pipeline_rf,open('pipe.pkl','wb'))
```

### Final model file
- model.py
```python
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

```

### Deployment of model

- We deploy our model using Flask framework
- test.py file

```python
import pandas as pd
import numpy as np
from flask import Flask, request, send_file, render_template
import pickle
import os

app = Flask(__name__)
model = pickle.load(open('pipe.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    file=request.files['file']
    data = pd.read_csv(file)

    col_names = data.columns.values.tolist()
    col_sorted = col_names.sort()
    list = ['date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    list_sort = list.sort()
    if(col_sorted == list_sort):
        temp = data['Temperature'].mean()
        hum = data['Humidity'].mean()
        light = data['Light'].mean()
        co2 = data['CO2'].mean()
        hr = data['HumidityRatio'].mean()

        data = data.fillna(value={
            'Temperature':temp,
            'Humidity':hum,
            'Light':light,
            'CO2':co2,
            'HumidityRatio':hr
        })

        new_file=data.drop(['date', 'HumidityRatio'], axis=1)

        prediction=model.predict(new_file)
        data['Occupancy']=prediction
        final = data.to_csv('roomoccupancy.csv', index=False)

        return render_template('index.html', prediction_text='Your output file is downloaded in your device named as "roomoccupancy.csv"')
    else:
        return render_template('index.html', prediction_text='Uploaded file has insufficient column')

if __name__ == "__main__":
    app.run(debug=True)

```
### HTML page 
```python
<!DOCTYPE html>
<html>

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <link rel="stylesheet" type="text/css" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <title>Room occupancy prediction</title>
</head>

<body>

    <div class="container">
        <br>
        <h2 class="text-center">Room occupancy precision</h2>
        <div class="col-lg-12 m-auto d-block">
            <form id="tryme" action="{{ url_for('predict')}}" name="form" method="POST" enctype="multipart/form-data" onsubmit="return validate(name, email, phone, image)">
                <div class="row jumbotron">
                    <!-- image -->

                    <div class="col-lg-6 form-group">
                        <label for="file">Upload Your csv file</label>
                        <input type="file" name="file" class="form-control-file" id="file" required>
                    </div>

                    <!-- submit -->

                    <div class="col-sm-6 form-group">
                        <button type="submit" name="submit" class="btn btn-primary float-centre" >Submit</button>
                    </div>
                </div>
            </form>
        </div>

        {{ prediction_text }}
    </div>
</body>

</html>
```
