import sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris=load_iris()
df=pd.DataFrame(iris['data'])
print(df)
print(iris['target_names'])
print(iris['feature_names'])
print(iris['target'])
X=df
y=iris['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
print("Correct prediction", accuracy_score(y_test,y_pred))
print("Wrong prediction", (1-accuracy_score(y_test,y_pred)))
y_testtrain=knn.predict(X_train)
cm1=confusion_matrix(y_train,y_testtrain)
print(cm1)
