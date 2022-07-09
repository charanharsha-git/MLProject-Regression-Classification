import pandas as pd
import numpy as np
from matplotlib.cm import rainbow
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
import app as a

from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier




def cat_var(x):
    dtype = pd.DataFrame(x.dtypes, columns=["DataType"])
    return dtype[dtype.DataType == "object"].index.to_list()

def x_n_y(data,var):
    X=data.drop(columns=[str(var)])
    Y=data[var]
    return X,Y

def train_test_split1(X,Y):
    return train_test_split(X, Y, test_size=0.2)

def preprocessing(x):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
    le = LabelEncoder()
    dtype = pd.DataFrame(x.dtypes, columns=["DataType"])
    cat_var = dtype[dtype.DataType == "object"].index.to_list()
    for i in cat_var:
        x[i].replace(to_replace=" ?",
                        value=x[i].mode()[0], inplace=True)
        label_encoding = le.fit_transform(x[i])
        x[i + "_label_encoding"] = label_encoding
    x = x.drop(columns=cat_var)
    x.dropna(inplace=True)

    return x

def models(x_train,y_train,x_test,y_test):
    classifiers = [GaussianNB(),
                   SVC(kernel='rbf', probability=True),
                   DecisionTreeClassifier(random_state=0),
                   RandomForestClassifier(n_estimators=100, random_state=0),
                   GradientBoostingClassifier(random_state=0),
                   KNeighborsClassifier(n_neighbors=5)
                   ]
    classifier_names = ["Gaussian Naive Bayes",
                        "Support Vector Classifier",
                        "Decision Tree Classifier",
                        "Random Forest Classifier",
                        "Gradient Boosting Classifier",
                        "K-Nearest Neighbours"]
    accuracies = []
    for i in range(len(classifiers)):
        classifier = classifiers[i]
        classifier.fit(x_train, y_train.astype('int'))
        y_pred = classifier.predict(x_test)
        print(np.unique(y_pred))
        accuracy = accuracy_score(y_test, y_pred) * 100
        accuracies.append(accuracy)
        print(accuracies)

    #class_plot=plt.plot(classifier_names,accuracies,kind='bar')
    #class_plot.savefig("classification.png")
    plt.figure(figsize=(13, 6))
    colors = rainbow(np.linspace(0, 1, len(classifiers)))
    barplot = plt.bar(classifier_names, accuracies, color=colors)
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=16)
    plt.xticks(fontsize=14)
    plt.xlabel("Classifiers", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.title("Plot for accuracy of all classifiers", fontsize=16)
    for i, bar in enumerate(barplot):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1,
                 bar.get_height() * 1.02,
                 s='{:.2f}%'.format(accuracies[i]),
                 fontsize=16)
    plt.savefig('static/classification.png')

    return accuracies,classifier_names
