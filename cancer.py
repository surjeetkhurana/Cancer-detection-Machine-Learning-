import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
#from sklearn.externals import joblib
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
import xgboost

#from pandas_profiling import ProfileReport
logreg=LogisticRegression()

data=pd.read_csv("cancer.csv")
data.drop(["Unnamed: 32"],axis="columns",inplace=True)
data.drop(["id"],axis="columns",inplace=True)
a=pd.get_dummies(data["diagnosis"])
cancer=pd.concat([data,a],axis="columns")
cancer.drop(["diagnosis","B"],axis="columns",inplace=True)
cancer.rename(columns={"M":"Malignant/Benign"},inplace=True)
y=cancer[["Malignant/Benign"]]
X=cancer.drop(["Malignant/Benign"],axis="columns")
print(X.shape[1])





data.info()
data.describe()


sns.set_style('whitegrid')
sns.lmplot('radius_mean' , 'texture_mean', hue = 'diagnosis', data = data, palette = 'coolwarm', size = 6, fit_reg = False )



# check if any null value is present
data.isnull().values.any()




## Here we will check the percentage of nan values present in each feature
## 1 -step make the list of features which has missing values
features_with_na=[features for features in data.columns if data[features].isnull().sum()>1]
## 2- step print the feature name and the percentage of missing values

for feature in features_with_na:
    print(feature, np.round(dataset[feature].isnull().mean(), 4),  ' % missing values')

#Since there are no missing values so we move forward




#Now we find numerical variables in the features
# list of numerical variables
numerical_features = [feature for feature in data.columns if data[feature].dtypes != 'O']

print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
data[numerical_features].head()


#Now we try to find the temporal variables
# list of variables that contain year information
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

year_feature




#Since there are no temporal variables we move forward



## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_feature=[feature for feature in numerical_features if len(data[feature].unique())<25 and feature not in year_feature+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# Correlation

#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")




data.corr()




diagnosis_map = {'M': 1, 'B': 0}


data['diagnosis'] = data['diagnosis'].map(diagnosis_map)




diagnosis_true_count = len(data.loc[data['diagnosis'] == 1])
diagnosis_false_count = len(data.loc[data['diagnosis'] == 0])

diagnosis_true_count,diagnosis_false_count

    
    
## Train Test Split


feature_columns = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
predicted_class = ['diagnosis']
    
    
    
X = data[feature_columns].values
y = data[predicted_class].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)



## Apply Algorithm


random_forest_model = RandomForestClassifier(random_state=10)

random_forest_model.fit(X_train, y_train.ravel())



predict_train_data = random_forest_model.predict(X_test)



print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))



## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}



#help
#https://github.com/krishnaik06/Diabetes-Prediction/blob/master/Diabetes_Prediction.ipynb

#dataset
#https://www.kaggle.com/code/buddhiniw/breast-cancer-prediction/data