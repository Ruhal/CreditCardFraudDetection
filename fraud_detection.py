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

#------------------------------------------------------------------------

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

#------------------------------------------------------------------------

# Prediction with no model, assuming all transactions are legit
predictions = np.zeros((credit_card.shape[0],), dtype=int)
predictions = pd.Categorical(predictions, categories=[0, 1])
conf_mat = sklm.confusion_matrix(credit_card['Class'], predictions)
print(conf_mat)
print(sklm.classification_report(credit_card['Class'], predictions, zero_division=False, digits=4))

#------------------------------------------------------------------------

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

#------------------------------------------------------------------------

# Create training and test set
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, train_size=0.80, test_size=0.2, random_state=123)
for train_index, test_index in sss.split(cc_sample, cc_sample['Class']):
    train_data = credit_card.iloc[train_index,:]
    test_data = credit_card.iloc[test_index,:]
    
train_data.shape
test_data.shape

#------------------------------------------------------------------------

