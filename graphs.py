import pandas as pd
import matplotlib.pyplot as plt

# Dataset
dataset = pd.read_csv("loan_data.csv", header=0)

bins_age = [18, 25, 35, 45, 55, 65, 75, 100]
labels_age = ['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76+']
dataset['age_group'] = pd.cut(dataset['person_age'], bins=bins_age, labels=labels_age, right=False)

bins_income = [0, 100000, 200000, 300000, 400000, 500000, 10000000]
labels_income = ['0-100000', '100000-200000', '200000-300000', '300000-400000', '400000-500000', '500000+']
dataset['income_dist'] = pd.cut(dataset['person_income'], bins=bins_income, labels=labels_income, right=False)

bins_emp_exp = [0, 10, 20, 30, 40, 50, 1000]
labels_emp_exp = ['0-10', '10-20', '20-30', '30-40', '40-50', '50+']
dataset['person_emp_exp'] = pd.cut(dataset['person_emp_exp'], bins=bins_emp_exp, labels=labels_emp_exp, right=False)

bins_lo_am = [0, 5000, 10000, 20000, 30000, 40000, 50000, 100000]
labels_lo_am = ['0-5000', '5000-10000', '1000-20000', '20000-30000', '30000-40000', '40000-50000', '50000+']
dataset['loan_amnt'] = pd.cut(dataset['loan_amnt'], bins=bins_lo_am, labels=labels_lo_am, right=False)

bins_intrate = [0, 4, 8, 12, 16, 20, 24, 100]
labels_intrate = ['0-4', '4-8', '8-12', '12-16', '16-20', '20-24', '24+']
dataset['loan_int_rate'] = pd.cut(dataset['loan_int_rate'], bins=bins_intrate, labels=labels_intrate, right=False)

bins_perc_income = [0, 0.1, 0.2, 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
labels_perc_income = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7' ,'0.7-0.8', '0.8-0.9', '0.9-1']
dataset['loan_percent_income'] = pd.cut(dataset['loan_percent_income'], bins=bins_perc_income, labels=labels_perc_income, right=False)

bins_perc_cred = [0, 4, 8, 12, 16, 20, 24, 100]
labels_perc_cred = ['0-4', '4-8', '8-12', '12-16', '16-20', '20-24', '24+']
dataset['cb_person_cred_hist_length'] = pd.cut(dataset['cb_person_cred_hist_length'], bins=bins_perc_cred, labels=labels_perc_cred, right=False)

bins_cscore = [0, 150, 300, 450, 600, 750, 900, 1000]
labels_cscore = ['0-150', '150-300', '300-450', '450-600', '600-750', '750-900', '900-1000']
dataset['credit_score'] = pd.cut(dataset['credit_score'], bins=bins_cscore, labels=labels_cscore, right=False)

age_group_counts = dataset['age_group'].value_counts(sort=False)
gender_dist = dataset['person_gender'].value_counts(sort=False)
person_edu = dataset['person_education'].value_counts(sort=False)
person_income = dataset['income_dist'].value_counts(sort=False)
person_emp_exp = dataset['person_emp_exp'].value_counts(sort=False)
person_home_ownership = dataset['person_home_ownership'].value_counts(sort=False)
loan_amnt = dataset['loan_amnt'].value_counts(sort=False)
loan_intent = dataset['loan_intent'].value_counts(sort=False)
loan_int_rate = dataset['loan_int_rate'].value_counts(sort=False)
loan_percent_income = dataset['loan_percent_income'].value_counts(sort=False)
cb_person_cred_hist_length = dataset['cb_person_cred_hist_length'].value_counts(sort=False)
credit_score = dataset['credit_score'].value_counts(sort=False)
previous_loan_defaults_on_file = dataset['previous_loan_defaults_on_file'].value_counts(sort=False)
loan_status = dataset['loan_status'].value_counts(sort=False)

# 4 by 4 graph
fig, axes = plt.subplots(4, 4, figsize=(24, 12)) 

# AGE
age_group_counts.plot(kind='bar', color='skyblue', edgecolor='black', ax=axes[0][0])
axes[0][0].set_title('Age Distribution of Persons', fontsize=16)
axes[0][0].set_xlabel('Age Groups', fontsize=12)
axes[0][0].set_ylabel('Number of Persons', fontsize=12)
axes[0][0].tick_params(axis='x', rotation=45)

# GENDER
gender_dist.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors, ax=axes[0][1])
axes[0][1].set_title('Gender Distribution of Persons', fontsize=16)
axes[0][1].set_ylabel('')  # Gereksiz etiketleri kaldırmak için

# EDUCATION
person_edu.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors, ax=axes[0][2])
axes[0][2].set_title('Education Distribution of Persons', fontsize=16)
axes[0][2].set_ylabel('')

# INCOME
person_income.plot(kind='bar', color='skyblue', edgecolor='black', ax=axes[0][3])
axes[0][3].set_title('Income Distribution of Persons', fontsize=16)
axes[0][3].set_xlabel('Incomes', fontsize=12)
axes[0][3].set_ylabel('Number of Persons', fontsize=12)
axes[0][3].tick_params(axis='x', rotation=45)

person_emp_exp.plot(kind='bar', color='skyblue', edgecolor='black', ax=axes[1][0])
axes[1][0].set_title('Professional Experience Distribution of Persons', fontsize=16)
axes[1][0].set_xlabel('Experiences', fontsize=12)
axes[1][0].set_ylabel('Number of Persons', fontsize=12)
axes[1][0].tick_params(axis='x', rotation=45)

person_home_ownership.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors, ax=axes[1][1])
axes[1][1].set_title('Home Ownership Distribution of Persons', fontsize=16)
axes[1][1].set_ylabel('')

loan_amnt.plot(kind='bar', color='skyblue', edgecolor='black', ax=axes[1][2])
axes[1][2].set_title('Loan Amount Distribution of Persons', fontsize=16)
axes[1][2].set_xlabel('Amounts', fontsize=12)
axes[1][2].set_ylabel('Number of Persons', fontsize=12)
axes[1][2].tick_params(axis='x', rotation=45)

loan_intent.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors, ax=axes[1][3])
axes[1][3].set_title('Intent Distribution of Persons', fontsize=16)
axes[1][3].set_ylabel('')

loan_int_rate.plot(kind='bar', color='skyblue', edgecolor='black', ax=axes[2][0])
axes[2][0].set_title('Loan Interest Distribution of Persons', fontsize=16)
axes[2][0].set_xlabel('Interest Rates', fontsize=12)
axes[2][0].set_ylabel('Number of Persons', fontsize=12)
axes[2][0].tick_params(axis='x', rotation=45)

loan_percent_income.plot(kind='bar', color='skyblue', edgecolor='black', ax=axes[2][1])
axes[2][1].set_title('Loan Percent Income Distribution of Persons', fontsize=16)
axes[2][1].set_xlabel('Percentages', fontsize=12)
axes[2][1].set_ylabel('Number of Persons', fontsize=12)
axes[2][1].tick_params(axis='x', rotation=45)

cb_person_cred_hist_length.plot(kind='bar', color='skyblue', edgecolor='black', ax=axes[2][2])
axes[2][2].set_title('Loan Amount Distribution of Persons', fontsize=16)
axes[2][2].set_xlabel('Amounts', fontsize=12)
axes[2][2].set_ylabel('Number of Persons', fontsize=12)
axes[2][2].tick_params(axis='x', rotation=45)

credit_score.plot(kind='bar', color='skyblue', edgecolor='black', ax=axes[2][3])
axes[2][3].set_title('Credit Score Distribution of Persons', fontsize=16)
axes[2][3].set_xlabel('Scores', fontsize=12)
axes[2][3].set_ylabel('Number of Persons', fontsize=12)
axes[2][3].tick_params(axis='x', rotation=45)

previous_loan_defaults_on_file.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors, ax=axes[3][0])
axes[3][0].set_title('Previous loans of Persons', fontsize=16)
axes[3][0].set_ylabel('')

loan_status.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors, ax=axes[3][1])
axes[3][1].set_title('Loan Status of Persons', fontsize=16)
axes[3][1].set_ylabel('')

#delete last 2 empty graphs
fig.delaxes(axes[3, 2])  
fig.delaxes(axes[3, 3])  

plt.tight_layout()
plt.show()