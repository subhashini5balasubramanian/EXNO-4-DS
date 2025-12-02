# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.
A
# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
 import pandas as pd
 import numpy as np
 import seaborn as sns
 from sklearn.model_selection import train_test_split
 from sklearn.neighbors import KNeighborsClassifier
 from sklearn.metrics import accuracy_score, confusion_matrix
 data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
 data
```
<img width="816" height="772" alt="image" src="https://github.com/user-attachments/assets/98f6e6c0-55e8-4e76-883e-fe896a051c0c" />

```
 data.isnull().sum()

```
<img width="242" height="458" alt="image" src="https://github.com/user-attachments/assets/ec3c2030-fd71-4e16-9ce5-5415539c424f" />

```
 missing=data[data.isnull().any(axis=1)]
 missing
```
<img width="809" height="747" alt="image" src="https://github.com/user-attachments/assets/e661666c-1dcd-4c71-bc3a-59b713dcde47" />

```
 data2=data.dropna(axis=0)
 data2
```
<img width="813" height="735" alt="image" src="https://github.com/user-attachments/assets/ef10b695-2681-41de-81a8-272684177d5e" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
 print(data2['SalStat'])
```
<img width="847" height="318" alt="image" src="https://github.com/user-attachments/assets/a6a5f052-15d5-4f3e-8a8a-4d5a148993fd" />

```
 sal2=data2['SalStat']
 dfs=pd.concat([sal,sal2],axis=1)
 dfs

```
<img width="345" height="384" alt="image" src="https://github.com/user-attachments/assets/0c861adb-11b7-4011-b6c9-2bd11c5806f1" />

```
 data2

```
<img width="833" height="558" alt="image" src="https://github.com/user-attachments/assets/7ca19039-aa0c-4223-aa02-bcb397c7cb1f" />

```

new_data=pd.get_dummies(data2, drop_first=True)
new_data

```
<img width="842" height="456" alt="image" src="https://github.com/user-attachments/assets/5271a366-d4b6-4651-9461-a85eb3098094" />

```
 columns_list=list(new_data.columns)
 print(columns_list)

```
<img width="843" height="42" alt="image" src="https://github.com/user-attachments/assets/c1a61ba2-e426-4863-977d-44d3e3e4289f" />

```
features=list(set(columns_list)-set(['SalStat']))
 print(features)
```

<img width="842" height="35" alt="image" src="https://github.com/user-attachments/assets/cfdb4cbd-b0ac-4b39-9c0c-137d91830b8f" />

```
y=new_data['SalStat'].values
 print(y)
```
<img width="213" height="29" alt="image" src="https://github.com/user-attachments/assets/daa92206-b41d-40fa-bdb7-9e96115a500d" />

```
x=new_data[features].values
 print(x)

```
<img width="385" height="132" alt="image" src="https://github.com/user-attachments/assets/f886d633-6266-4495-84ea-d6512b5f3117" />

```
 train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
 KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

 KNN_classifier.fit(train_x,train_y)
 prediction=KNN_classifier.predict(test_x)
 confusionMatrix=confusion_matrix(test_y, prediction)
 print(confusionMatrix)

```
<img width="183" height="47" alt="image" src="https://github.com/user-attachments/assets/20c46697-ac66-41f7-95be-b501c8b46730" />

```
 accuracy_score=accuracy_score(test_y,prediction)
 print(accuracy_score)

```
<img width="219" height="29" alt="image" src="https://github.com/user-attachments/assets/24e83308-60af-47ab-98db-caa55ca0baa9" />

```
 print("Misclassified Samples : %d" % (test_y !=prediction).sum())

```
<img width="309" height="28" alt="image" src="https://github.com/user-attachments/assets/ae93bf8f-4169-40be-9335-a82d85aec48d" />

```
 data.shape

```
<img width="152" height="45" alt="image" src="https://github.com/user-attachments/assets/546b103a-3eea-4236-9f1c-df32c5784433" />

```
import pandas as pd
 from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
 data={
 'Feature1': [1,2,3,4,5],
 'Feature2': ['A','B','C','A','B'],
 'Feature3': [0,1,1,0,1],
 'Target'  : [0,1,1,0,1]
 }
 df=pd.DataFrame(data)
 x=df[['Feature1','Feature3']]
 y=df[['Target']]
 selector=SelectKBest(score_func=mutual_info_classif,k=1)
 x_new=selector.fit_transform(x,y)
 selected_feature_indices=selector.get_support(indices=True)
 selected_features=x.columns[selected_feature_indices]
 print("Selected Features:")
 print(selected_features)

```
<img width="842" height="93" alt="image" src="https://github.com/user-attachments/assets/d024a682-d8e5-4647-8fe3-17c195358d19" />

```
 import pandas as pd
 import numpy as np
 from scipy.stats import chi2_contingency
 import seaborn as sns
 tips=sns.load_dataset('tips')
 tips.head()
```
<img width="456" height="184" alt="image" src="https://github.com/user-attachments/assets/ddffa0d5-eca7-4cfc-93e1-b66cfe7e4d87" />

```
 tips.time.unique()

```
<img width="378" height="44" alt="image" src="https://github.com/user-attachments/assets/e3972c4c-34bc-4958-bda4-6e9a9af70085" />

```
 contingency_table=pd.crosstab(tips['sex'],tips['time'])
 print(contingency_table)

```
<img width="227" height="75" alt="image" src="https://github.com/user-attachments/assets/127cabb9-935b-4ce8-b5ad-228c6ee0afcb" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")

```
<img width="358" height="38" alt="image" src="https://github.com/user-attachments/assets/0b051ed3-d51d-485f-8228-508802f67381" />





      
# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
 save the data to a file is been executed

       
