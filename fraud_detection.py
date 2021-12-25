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
plt.pie(credit_card['Class'].value_counts()/credit_card['Class'].value_counts().sum(),
        labels = ["Legit", "Fraud"], colors = ['b','r'], autopct='%.2f')
plt.title("Pie chart of credit card transactions")
plt.show()

#------------------------------------------------------------------------

# Prediction with no model
predictions = np.zeros((credit_card.shape[0],), dtype=int)
predictions = pd.Categorical(predictions, categories=[0, 1])
sklm.confusion_matrix(credit_card['Class'], predictions)
sklm.confusion_matrix()