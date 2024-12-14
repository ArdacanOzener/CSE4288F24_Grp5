import pandas as pd
from sklearn.model_selection import train_test_split

#create dataset
dataset = pd.read_csv("loan_data.csv", header= 0)

#print(dataset.head())

#split into 2 different datasets according to the 'loan_status' infos
loan_status_yes = dataset[dataset['loan_status'] == 1]
loan_status_no = dataset[dataset['loan_status'] == 0]


#print("Column names:", list(dataset.columns))

num_of_yes = loan_status_yes.shape[0]
num_of_no = loan_status_no.shape[0]

#Number of loan status = 1 indices (35000)
yes_indices = list(range(num_of_yes))
train_yes_index, test_yes_index = train_test_split(yes_indices, test_size=0.2, random_state=42)
train_yes_index, validation_yes_index = train_test_split(train_yes_index, test_size = 1/8, random_state= 42)

#Number of loan status = 0 indices (10000)
no_indices = list(range(num_of_no))
train_no_index, test_no_index = train_test_split(no_indices, test_size=0.2, random_state=42)
train_no_index, validation_no_index = train_test_split(train_no_index, test_size = 1/8, random_state= 42)

#get appropriate row numbers from dataset according to the indices
#for loan_status = 1
train_yes = loan_status_yes.iloc[train_yes_index]
validation_yes = loan_status_yes.iloc[validation_yes_index]
test_yes = loan_status_yes.iloc[test_yes_index]

#for loan_status = 0
train_no = loan_status_no.iloc[train_no_index]
validation_no = loan_status_no.iloc[validation_no_index]
test_no = loan_status_no.iloc[test_no_index]

#merging operations
train_data = pd.concat([train_yes, train_no],ignore_index = True)
validation_data = pd.concat([validation_yes, validation_no],ignore_index = True)
test_data = pd.concat([test_yes, test_no],ignore_index = True)

print(len(train_data))
