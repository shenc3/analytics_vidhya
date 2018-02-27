import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

feats = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']
target = ['Loan_Status']
all_data = pd.concat([train[feats], test[feats]], axis=0)
all_data.reset_index(drop=True, inplace=True)

# 各属性空值的数量
# all_data.isnull().sum(axis=0).to_frame()
na_info = pd.concat([
    all_data.isnull().sum(axis=0).to_frame('isna'),
    all_data.dtypes.to_frame('dtypes')
    ], axis=1)

'''
[out]:
                       isna   dtypes
    Loan_ID               0   object
    Gender               13   object
    Married               3   object
    Dependents           15   object
    Education             0   object
    Self_Employed        32   object
    ApplicantIncome       0    int64
    CoapplicantIncome     0  float64
    LoanAmount           22  float64
    Loan_Amount_Term     14  float64
    Credit_History       50  float64
    Property_Area         0   object
    Loan_Status           0   object
'''

# object类型用众数填充，float64类型用mean填充
for col in na_info[na_info['isna']>0].index.values:
    if na_info.loc[col]['dtypes'] == 'object':
        tmp = all_data[col].fillna('Null')
        le = LabelEncoder().fit(tmp)
        all_data.loc[all_data[col].isnull(), col] = le.classes_[np.argmax(np.bincount(le.transform(tmp)))]
    if na_info.loc[col]['dtypes'] == 'float64':
        imp = Imputer(strategy='mean')
        all_data[[col]] = imp.fit_transform(all_data[[col]])
print(all_data.isnull().sum(axis=0))

# 
unique_info = pd.concat([
    all_data.apply(lambda x: np.unique(x).shape[0], axis=0).to_frame('n_unique'),
    all_data.dtypes.to_frame('dtypes')
    ], axis=1)
feats_obj = unique_info[unique_info['dtypes']=='object'].index.values
feats_num = list(set(feats) - set(feats_obj))

all_data2 = pd.concat([
    pd.get_dummies(all_data[feats_obj]), all_data[feats_num]
    ], axis=1)

# 生成训练数据集
X = all_data2.iloc[:len(train)]
y = le.fit_transform(train['Loan_Status'])
test_set = all_data2.iloc[len(train):]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=9)

# 训练 RandomForest
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred1 = clf.predict(X_test)

# 评估结果
print(classification_report(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))

'''
[out]:
             precision    recall  f1-score   support

          0       0.64      0.49      0.56        65
          1       0.76      0.85      0.80       120

avg / total       0.71      0.72      0.71       185

# confusion_matrix
array([[ 32,  33],
       [ 18, 102]], dtype=int64)
'''

# 训练
feats_obj2 = list(set(all_data2.columns) - set(feats_num))  # 除了数值特征之外的特征
pipeline = Pipeline([
    ('scaler', StandardScaler())
    ])  # pipeline用来预处理数据

# 将数值特征标准化，之后再训练
clf = LogisticRegression(penalty='l1', C=0.1)
clf.fit(np.hstack([
    X_train[feats_obj2].values,
    pipeline.fit_transform(X_train[feats_num]
    )]), y_train)
y_pred2 = clf.predict(np.hstack([
    X_test[feats_obj2].values,
    pipeline.fit_transform(X_test[feats_num])
    ]))

# 评估结果
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))

'''
[out]:
             precision    recall  f1-score   support

          0       0.93      0.38      0.54        65
          1       0.75      0.98      0.85       120

avg / total       0.81      0.77      0.74       185

confusion_matrix
array([[ 25,  40],
       [  2, 118]], dtype=int64)
'''
