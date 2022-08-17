import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import shuffle

iris = pd.read_csv("iris.csv")
print(iris.drop(['id', 'variety'], axis=1).describe())
print(iris['variety'].value_counts())
sns.pairplot(iris.drop(['id'], axis=1), hue='variety')

X = iris.drop(['variety', 'id'], axis=1)
y = iris['variety']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=100)

model = SVC(C=1, kernel='rbf', tol=0.001)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))
print('\n')
print('Accuracy score is: ', accuracy_score(y_test, pred))

X = iris.drop('variety', axis=1)
y = iris['variety']
print('Before shuffle: ', y[0:20])
X, y = shuffle(X, y, random_state=0)
print("After shuffle: ", y[0:20])

X = iris.drop(['variety', 'id'], axis=1)
y = iris['variety']
param_grid = {'C': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
grid = GridSearchCV(estimator=SVC(), param_grid=param_grid, scoring='accuracy',
                    cv=KFold(n_splits=3, shuffle=True, random_state=0), verbose=1, refit=True)


grid.fit(X, y)
print(grid)

print(grid.best_params_)
print(grid.best_estimator_)
print('Mean cross-validated score of the best_estimator: ', grid.best_score_)
print('The number of cross-validation splits (folds/iterations): ', grid.n_splits_)
