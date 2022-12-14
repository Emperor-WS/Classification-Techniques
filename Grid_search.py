#Importing libraries

from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing DS
dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

#Spilitting the dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)

#Feature Scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Fiting Kernel SVM to the training set
classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(x_train, y_train)

#Predicitng The Test Set Results
y_pred = classifier.predict(x_test)

#Making The Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

#Applying k-Fold cross validation
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)

#Applying Grid-Search
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ["linear"]},
              {'C': [1, 10, 100, 1000], 'kernel': ["rbf"], "gamma": [0.5, 0.1, 0.01, 0.001]}]
grid_search = GridSearchCV(
    estimator=classifier, param_grid=parameters, scoring="accuracy", cv=10)
grid_search = grid_search.fit(x_train, y_train)
best_accurancy = grid_search.best_score_
best_parameters = grid_search.best_params_

# Visualising the Training set results
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Kernel SVM  (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
