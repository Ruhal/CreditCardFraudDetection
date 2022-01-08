import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sklm

# Import Credit Card dataset, source: https://www.kaggle.com/mlg-ulb/creditcardfraud
credit_card = pd.read_csv(r'C:\Users\Ruhal\Documents\creditcard.csv')

# Convert Class to a categorical variable
credit_card['Class'] = pd.Categorical(credit_card['Class'], categories=[0, 1])

# get an overview of the data
credit_card.describe()

# The number of missing values in our dataset
credit_card.isna().sum().sum()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# distribution of fraudulent and legitimate transactions in the dataset
credit_card['Class'].value_counts()

# percentage of fraud and legit transactions in dataset
credit_card['Class'].value_counts()/credit_card['Class'].value_counts().sum()

# Pie chart of fraud and legit transactions
plt.figure(dpi=300)
plt.pie(credit_card['Class'].value_counts()/credit_card['Class'].value_counts().sum(),
        labels = ["Legit", "Fraud"], colors = ['b','r'], autopct='%.2f')
plt.title("Pie chart of credit card transactions")
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Prediction with no model, assuming all transactions are legit
predictions = np.zeros((credit_card.shape[0],), dtype=int)
predictions = pd.Categorical(predictions, categories=[0, 1])
conf_mat = sklm.confusion_matrix(credit_card['Class'], predictions)
print(conf_mat)
print(sklm.classification_report(credit_card['Class'], predictions, zero_division=False, digits=4))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Random sample with replacement with fixed seed
cc_sample = credit_card.sample(frac=0.1, replace=True, random_state=1)

cc_sample['Class'].value_counts()

plt.figure(dpi=300)
plt.title("Scatter graph between V1 and V2")
plt.scatter(cc_sample[cc_sample['Class']==0]['V1'], cc_sample[cc_sample['Class']==0]['V2'], s=3, c='cyan')
plt.scatter(cc_sample[cc_sample['Class']==1]['V1'], cc_sample[cc_sample['Class']==1]['V2'], s=3, c='r')
plt.legend(['0', '1'], title="Class")
plt.xlabel("V1")
plt.ylabel("V2")

plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create training and test set
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, train_size=0.80, test_size=0.2, random_state=123) # 80% training and 20% test
for train_index, test_index in sss.split(cc_sample, cc_sample['Class']):
    train_data = credit_card.iloc[train_index,:]
    test_data = credit_card.iloc[test_index,:]
    
train_data.shape
test_data.shape

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Random Over-Sampling (ROS) - will over sample the minority class which is fraud

train_data['Class'].value_counts()

n_legit = train_data['Class'].value_counts()[0]
new_legit_fraction = 1/2
new_n_total = n_legit/new_legit_fraction

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=123, sampling_strategy=(1-new_legit_fraction)/new_legit_fraction)
oversampled_credit, y_resampled = ros.fit_resample(train_data, train_data['Class'])

oversampled_credit['Class'].value_counts() # dist of fraud and legit is equal

plt.figure(dpi=300)
plt.title("Scatter graph between V1 and V2 (ROS)")

def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

plt.scatter(oversampled_credit[oversampled_credit['Class']==0]['V1'], oversampled_credit[oversampled_credit['Class']==0]['V2'], s=3, c='cyan')
plt.scatter(rand_jitter(oversampled_credit[oversampled_credit['Class']==1]['V1']), rand_jitter(oversampled_credit[oversampled_credit['Class']==1]['V2']), s=3, c='r')
plt.legend(['0', '1'], title="Class")
plt.xlabel("V1")
plt.ylabel("V2")
plt.show()

# ROS creates duplicates of fraud cases so added jitter allows you to see there are more fraud cases now


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Random Under-Sampling (RUS)

train_data['Class'].value_counts()
n_fraud = train_data['Class'].value_counts()[1]
new_fraud_fraction = 1/2
new_n_total = n_fraud/new_legit_fraction

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=123, sampling_strategy=new_fraud_fraction/(1-new_fraud_fraction))
undersampled_credit, y_resampled = rus.fit_resample(train_data, train_data['Class'])

undersampled_credit['Class'].value_counts() 


plt.figure(dpi=300)
plt.title("Scatter graph between V1 and V2 (RUS)")
plt.scatter(undersampled_credit[undersampled_credit['Class']==0]['V1'], undersampled_credit[undersampled_credit['Class']==0]['V2'], s=3, c='cyan')
plt.scatter(rand_jitter(undersampled_credit[undersampled_credit['Class']==1]['V1']), rand_jitter(undersampled_credit[undersampled_credit['Class']==1]['V2']), s=3, c='r')
plt.legend(['0', '1'], title="Class")
plt.xlabel("V1")
plt.ylabel("V2")
plt.show()

# Can see equal distrubtion of fraud and legit

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Both ROS and RUS combined

n_new = len(train_data)
new_fraud_fraction = 1/3
over = RandomOverSampler(random_state=123, sampling_strategy=new_fraud_fraction/(1-new_fraud_fraction))
X, y = over.fit_resample(train_data, train_data['Class'])
under = RandomUnderSampler(random_state=123)
sampled_credit, y = under.fit_resample(X, y)

sampled_credit['Class'].value_counts() # equal dist of fraud and legit

plt.figure(dpi=300)
plt.title("Scatter graph between V1 and V2 (ROS and RUS)")

def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

plt.scatter(sampled_credit[sampled_credit['Class']==0]['V1'], sampled_credit[sampled_credit['Class']==0]['V2'], s=3, c='cyan')
plt.scatter(rand_jitter(sampled_credit[sampled_credit['Class']==1]['V1']), rand_jitter(sampled_credit[sampled_credit['Class']==1]['V2']), s=3, c='r')
plt.legend(['0', '1'], title="Class")
plt.xlabel("V1")
plt.ylabel("V2")
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Balance dataset with SMOTE method
from imblearn.over_sampling import SMOTE 

train_data['Class'].value_counts()
new_legit_fraction = 0.6

sm = SMOTE(random_state=123, sampling_strategy=(1-new_legit_fraction)/new_legit_fraction)
SMOTE_credit, y_resampled = sm.fit_resample(train_data.iloc[:,1:], train_data['Class']) # removed Time feature from sample

SMOTE_credit['Class'].value_counts() # 60% of data now legit, 40% fraud

plt.figure(dpi=300)
plt.title("Scatter graph between V1 and V2 (SMOTE)")
plt.scatter(SMOTE_credit[SMOTE_credit['Class']==0]['V1'], SMOTE_credit[SMOTE_credit['Class']==0]['V2'], s=3, c='cyan')
plt.scatter(SMOTE_credit[SMOTE_credit['Class']==1]['V1'], SMOTE_credit[SMOTE_credit['Class']==1]['V2'], s=3, c='r')
plt.legend(['0', '1'], title="Class")
plt.xlabel("V1")
plt.ylabel("V2")
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Decision Tree using SMOTE data

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus

clf = DecisionTreeClassifier(random_state=1).fit(SMOTE_credit.drop(['Class'], axis=1), SMOTE_credit['Class']) # create the decision tree using SMOTE sample

out = StringIO()
dt_target_names  = SMOTE_credit.drop(['Class'], axis=1).columns
export_graphviz(clf, out_file=out,  
                     class_names=dt_target_names, label = 'none', filled=True, rounded=True, special_characters=True)

graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png()) # require GraphViz to display the decision tree

# Predict fraudulent cases
Prediction_values = clf.predict(test_data.drop(['Class','Time'], axis=1)) # predict using decision tree and test data
unique, counts = np.unique(Prediction_values, return_counts=True)
np.asarray((unique, counts)).T

# Build confusion matrix
Model_conf_mat = sklm.confusion_matrix(test_data['Class'], Prediction_values)
print(Model_conf_mat)

print(sklm.classification_report(test_data['Class'], Prediction_values, zero_division=False, digits=4))
print("Accuracy:", sklm.accuracy_score(test_data['Class'], Prediction_values))
# We have a good accuracy of around from SMOTE sample 0.998

clf_SMOTE = clf

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Decision Tree on unbalanced training data

train_data = train_data.drop(['Time'], axis=1)
clf = DecisionTreeClassifier(random_state=1).fit(train_data.drop(['Class'], axis=1), train_data['Class']) # create the decision tree using train_data sample

out = StringIO()
dt_target_names  = train_data.drop(['Class'], axis=1).columns
export_graphviz(clf, out_file=out,  
                     class_names=dt_target_names, label = 'none', filled=True, rounded=True, special_characters=True)

graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png()) # require GraphViz to display the decision tree

# Predict fraudulent cases
Prediction_values = clf.predict(test_data.drop(['Class','Time'], axis=1)) # predict using decision tree and test data
unique, counts = np.unique(Prediction_values, return_counts=True)
np.asarray((unique, counts)).T

# Build confusion matrix
Model_conf_mat = sklm.confusion_matrix(test_data['Class'], Prediction_values)
print(Model_conf_mat)

print(sklm.classification_report(test_data['Class'], Prediction_values, zero_division=False, digits=4))
print("Accuracy:", sklm.accuracy_score(test_data['Class'], Prediction_values))

# Accuracy from train data is better at around 0.999

clf_train = clf
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Predict for the whole credit_card data set using SMOTE sample prediction

# Predict fraudulent cases
Prediction_values = clf_SMOTE.predict(credit_card.drop(['Class','Time'], axis=1)) # predict using decision tree and test data
unique, counts = np.unique(Prediction_values, return_counts=True)
np.asarray((unique, counts)).T

# Build confusion matrix
Model_conf_mat = sklm.confusion_matrix(credit_card['Class'], Prediction_values)
print(Model_conf_mat)

print(sklm.classification_report(credit_card['Class'], Prediction_values, zero_division=False, digits=4))
print("Accuracy:", sklm.accuracy_score(credit_card['Class'], Prediction_values))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Predict for the whole credit_card data set using train_data sample prediction

# Predict fraudulent cases
Prediction_values = clf_train.predict(credit_card.drop(['Class','Time'], axis=1)) # predict using decision tree and test data
unique, counts = np.unique(Prediction_values, return_counts=True)
np.asarray((unique, counts)).T

# Build confusion matrix
Model_conf_mat = sklm.confusion_matrix(credit_card['Class'], Prediction_values)
print(Model_conf_mat)

print(sklm.classification_report(credit_card['Class'], Prediction_values, zero_division=False, digits=4))
print("Accuracy:", sklm.accuracy_score(credit_card['Class'], Prediction_values))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Using the two sections above we can compare the predictions based on the balanced SMOTE sample
# and the prediction based on the unbalanced training data to see which is more accurate using this model






























