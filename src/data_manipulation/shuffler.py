import pandas as pd

training_data_path = 'datasets/training/training_data_ID.csv'
validation_data_path = 'datasets/validation/validation_data_ID.csv'
test_data_path = 'datasets/test/test_data_ID.csv'


training_data = pd.read_csv(training_data_path)
validation_data = pd.read_csv(validation_data_path)
test_data = pd.read_csv(test_data_path)

#shuffle training data
training_data = training_data.sample(frac=1)

#shuffle validation data
validation_data = validation_data.sample(frac=1)

#shuffle test data
test_data = test_data.sample(frac=1)

training_data.to_csv('datasets/training/training_data_ID.csv', index=False)
validation_data.to_csv('datasets/validation/validation_data_ID.csv', index=False)
test_data.to_csv('datasets/test/test_data_ID.csv', index=False)




