IRIS FLOWER CLASSIFICATION
using Support Vector Machine, Logistic Regression, Decision Tree Classifier

Support Vector Machines. SVM works by finding the optimal hyperplane that best separates data points belonging to different classes in a given feature space.
Logistic Regression- a statistical method used for binary classification tasks, where the output variable can take only two possible values, usually encoded as 0 and 1. Despite its name, logistic regression is a classification algorithm, not a regression algorithm.
Decision Tree Classifer- The Decision Tree Classifier is a popular machine learning algorithm used for classification tasks. It builds a decision tree model based on the training data, where each internal node represents a feature or attribute, each branch represents a decision rule, and each leaf node represents the outcome or class label

pandas (import pandas as pd): Pandas is a powerful library for data manipulation and analysis in Python. It provides data structures like DataFrame and Series, which allow you to work with structured data easily.

numpy (import numpy as np): NumPy is a fundamental package for scientific computing with Python. It provides support for arrays, matrices, and mathematical functions, making it essential for numerical operations and computations.

seaborn (import seaborn as sns): Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics. 
Seaborn simplifies the process of creating common visualization types such as scatter plots, bar plots, and histograms.

matplotlib.pyplot (import matplotlib.pyplot as plt): Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. The pyplot module provides a MATLAB-like interface for creating plots and visualizations. 
It is commonly used in combination with other libraries like NumPy and pandas for data visualization tasks.

%matplotlib inline: This is a magic command in Jupyter Notebook or JupyterLab environments. It allows matplotlib plots to be displayed directly within the notebook cells, enabling inline plotting without the need to call plt.show()

1- LOADING THE DATA
Reads the Iris dataset from a CSV file into a pandas DataFrame named df. The DataFrame consists of five columns: 'Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', and 'Class Labels'. 

2- VISUALIZE THE DATA

Prior using seaborn, we convert the data in columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'] to  numeric.
For this we use the pd.to_numeric() function.
Then we create a pairplot using seaborn.
The hue parameter is set to 'Class Labels', which means that the data points in the pairplot will be colored based on the values in the 'Class Labels' column.

3- SEPARATE THE INPUT COLUMNS AND OUTPUT COLUMNS

data = df.values
This line extracts the value from the dataframe df and assigns them to the variable data.
Data is a numpy array containing all the values from the dataframe.

X=data[:,0:4]
This line selects the feature columns from the data array and assigns them to the variable 

Y=data[:,4]
This line selects the target variable column from the data.

In summary, after executing these lines of code, you will have the feature matrix X, containing the feature variables (Sepal Length, Sepal Width, Petal Length, Petal Width), and the target array Y, containing the target variable (Class Labels), ready for use in machine learning algorithms.

4- SPLITTING THE DATA INTO TRAINING AND TESTING

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
print(Y_train)

To split the data into training and testing sets, you can use the train_test_split function from the sklearn.model_selection module. This function splits the dataset into two subsets: one for training the model and the other for testing its performance.

X_train: Training set features
X_test: Testing set features
Y_train: Training set target values
Y_test: Testing set target values

5- APPLYING SUPPORT VECTOR MACHINE ALGORITHM

from sklearn.svm import SVC
This line imports the Support Vector Classifier (SVC) class from the scikit-learn library. 

model_svc = SVC()
This line initializes an SVC model with default hyperparameters.

model_svc.fit(X_train,Y_train)
This line trains the SVC model using the training data. The fit() method fits the model to the training data by learning the patterns in the features (X_train) and their corresponding target values (Y_train).

prediction1=model_svc.predict(X_test)
This line makes predictions on the testing data (X_test) using the trained SVC model. The predict() method predicts the class labels for the input data based on the learned patterns from the training data.

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, prediction1) * 100)
This calculates accuracy of the SVC models preidctions by comparing the preicted class labels('prediction1) with the actual class labels from the testing data('Y_test')
he accuracy_score() function from scikit-learn's metrics module is used for this purpose. The accuracy score is then multiplied by 100 to express it as a percentage.

6- APPLYING LOGISTIC REGRESSION MODEL

from sklearn.linear_model import LogisticRegression
This line imports the LogisticRegression class from the scikit-learn library.

model_LR = LogisticRegression(max_iter=1000)
This line initializes a Logistic Regression model (model_LR) with a maximum number of iterations (max_iter) set to 1000. By setting max_iter to 1000, you are allowing the algorithm to run for more iterations before convergence.

model_LR.fit(X_train, Y_train)
This line trains the Logistic Regression model (model_LR) using the training data (X_train and Y_train). The fit() method fits the model to the training data by learning the patterns in the features (X_train) and their corresponding target values (Y_train).

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, prediction2) * 100)
This code calculates the accuracy of the Logistic Regression model's predictions by comparing the predicted class labels (prediction2)

7- APPLY DECISION TREE CLASSIFIER MODEL

Same steps as the above.We get 100% accuracy.

8- A Detailed Classification Report