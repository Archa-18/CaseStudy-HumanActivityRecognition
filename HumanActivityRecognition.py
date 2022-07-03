import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm

data = pd.read_csv('train.csv')

lr = LogisticRegression(random_state=0)
rf = RandomForestClassifier(random_state=1)
dt = DecisionTreeClassifier()
gb = GradientBoostingClassifier(n_estimators=10)
sv = svm.SVC()

data = pd.read_csv('train.csv')

x = data.drop(['Activity', 'subject'], axis=1)
y = data['Activity'].astype(object)

le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, train_size=0.3)

lr.fit(x_train, y_train)
rf.fit(x_train, y_train)
dt.fit(x_train, y_train)
gb.fit(x_train, y_train)
sv.fit(x_train, y_train)

lr_predict = lr.predict(x_test)
rf_predict = rf.predict(x_test)
dt_predict = dt.predict(x_test)
gb_predict = gb.predict(x_test)
sv_predict = sv.predict(x_test)

print('LogisticRegression', accuracy_score(y_test, lr_predict))
print('RandomForest', accuracy_score(y_test, rf_predict))
print('DecisionTree', accuracy_score(y_test, dt_predict))
print('GradientBoostingClassifier', accuracy_score(y_test, gb_predict))
print('SVM', accuracy_score(y_test, sv_predict))

#LogisticRegression 0.9714396735962697
#RandomForest 0.9677482028366039
#DecisionTree 0.914319020788809
#GradientBoostingClassifier 0.9411307557800661
#SVM 0.9638624441422188