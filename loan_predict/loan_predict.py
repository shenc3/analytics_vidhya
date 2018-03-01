import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.metrics import precision_recall_curve, roc_curve, auc
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

# 对分类变量进行one-hot编码，然后与连续变量构成新的数据集
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
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)
y_pred1 = rf.predict_proba(X_test)[:, 1]

# 评估结果
print(classification_report(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))

# 从随机森林中得到的特征重要性分数
feats_imp = pd.DataFrame(np.hstack([
  X_train.columns.values[:, None],
  clf.feature_importances_[:, None]
  ]), columns=['feature', 'score']).sort_values(by='score', ascending=False)

feats2use = feats_imp.loc[feats_imp['score'] > 0.04, 'feature'].values
X_train1 = X[feats2use]
X_test1 = test_set[feats2use]

'''
clf = RandomForestClassifier(n_estimators=100, oob_score=True)
clf.fit(X_train1, y)
y_pred1 = clf.predict(X_test1)

'''

cross_val_score(clf, X, y)


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

# ---------------------------------------------------------
# 正常逻辑回归
# 训练
lg = LogisticRegression(penalty='l2', C=1)
lg.fit(X_train, y_train)
y_pred2 = lg.predict_proba(X_test)[:,1]


# 评估结果
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))

# 交叉验证的结果
clf = LogisticRegression(penalty='l2', C=10)
clf.fit(X, y)
np.mean(cross_val_score(
  clf, X_train1, y,
  cv=KFold(n_splits=5, shuffle=True, random_state=0)))

submission = pd.DataFrame(
  clf.predict(test_set),
  index=test['Loan_ID'],
  columns=['Loan_Status']).reset_index()
submission['Loan_Status'] = submission['Loan_Status'].map(lambda x : {0:'N',1:'Y'}.get(x))

submission.to_csv('submission.csv', index=False)

# ---------------------------------------------------------
# 将数值特征标准化，之后再训练
'''
feats_obj2 = list(set(all_data2.columns) - set(feats_num))  # 除了数值特征之外的特征
pipeline = Pipeline([
    ('scaler', StandardScaler())
    ])  # pipeline用来预处理数据
X_train2 = np.hstack([
    X_train[feats_obj2].values,
    pipeline.fit_transform(X_train[feats_num]
    )])
X_test2 = np.hstack([
    X_test[feats_obj2].values,
    pipeline.fit_transform(X_test[feats_num])
    ])

# 训练
clf = LogisticRegression(penalty='l1', C=0.1)
clf.fit(X_train2, y_train)
y_pred2 = clf.predict(X_test2)

# 评估结果
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
'''

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

# 画出分类器的PRC曲线
def plot_prc(y_true, y_proba_pred, ax, legend):
    precision, recall, threshold = precision_recall_curve(
        y_true, y_proba_pred, pos_label=1, sample_weight=None)
    ax.plot(recall, precision)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision Recall Curve')
    ax.legend([legend,])

fig, ax = plt.subplots(figsize=(8, 8))
plot_prc(y_test, y_pred1, ax, 'RandomForest')
plot_prc(y_test, y_pred2, ax, 'LogisticRegression')
ax.plot([0, 1], [0,1], '--')


# 画出分类器的ROC曲线
def plot_roc():
    
