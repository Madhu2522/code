import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
df=pd.read_csv('kddcup99_csv.csv')
df.head()
df.columns
df.info()
df.shape
df.isna().sum()
print('Data set:')
for col_name in df.columns:
    if df[col_name].dtypes == 'object' :
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

print()
print('Distribution of categories in service:')
print(df['service'].value_counts().sort_values(ascending=False).head())
from sklearn import preprocessing
lab=preprocessing.LabelEncoder()
df['protocol_type']=lab.fit_transform(df['protocol_type'])
df['service']=lab.fit_transform(df['service'])
df['flag']=lab.fit_transform(df['flag'])
df.head()
df.info()
df1=df['label']
print('Label distribution Training set:')
print(df['label'].value_counts())
newdf=df1.replace({'normal':0,'smurf':1,'neptune':1,'back':1,'satan':2,'ipsweep':2,'portsweep':2,'warezclient': 2,'teardrop': 1,
                  'pod': 1,'nmap' : 2,'guess_passwd': 2,'buffer_overflow': 2,'land': 1,'warezmaster': 2,'imap': 2,'rootkit': 2,
                  'loadmodule': 2,'ftp_write': 2,'multihop': 2,'phf': 2,'perl': 2,'spy': 2})
print(newdf.head())
#newdf.to_csv('label.csv')
df['label'] = newdf
df.head()
df.info()
#df.to_csv('New_Data.csv')
data = pd.read_csv("New_Data.csv")
sns.countplot(x='label',data=data)
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
X = data.iloc[:,data.columns!='label'] 
y = data.iloc[:,data.columns=='label']  
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
bestfeatures = SelectKBest(score_func=chi2, k=17)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  
featureScores.nlargest(17,'Score') 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
xtrain, xtest, ytrain, ytest=train_test_split(X, y, test_size=0.33)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(xtrain,ytrain)
predic=rf.predict(xtest)
acc1=accuracy_score(predic,ytest)
acc1
clf=classification_report(predic,ytest)
print(clf)
from sklearn import svm
sv=svm.LinearSVC()
sv.fit(xtrain,ytrain)
predic1=sv.predict(xtest)
acc2=accuracy_score(predic1,ytest)
acc2
clf1=classification_report(predic1,ytest)
print(clf1)
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(xtrain,ytrain)
predic2=lg.predict(xtest)
acc3=accuracy_score(predic2,ytest)
acc3
clf2=classification_report(predic2,ytest)
print(clf2)
import matplotlib.pyplot as plt; plt.rcdefaults()
objects = ('Random Forest','Support Vector','LogisticRegression')
y_pos = np.arange(len(objects))
performance = [acc1,acc2,acc3]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('SVM vs Decision Tree')
plt.show()