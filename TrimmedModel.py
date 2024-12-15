import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

df_with_outliers = pd.read_csv("loan_data.csv", header=0)
    
def arrange_features(df):
    df['person_education'].replace({
    'High School': 0,
    'Associate': 0.25,
    'Bachelor': 0.5,
    'Master': 0.75,
    'Doctorate':1
    }, inplace=True)
    # Binary Encoding for person_gender
    df['person_gender'] = df['person_gender'].map({'female': 0, 'male': 1})
    # Binary Encoding for previous_loan_defaults_on_file
    df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})
    df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'], drop_first=True)
    df['person_home_ownership_OTHER'] = df['person_home_ownership_OTHER'].map({False: 0, True: 1})
    df['person_home_ownership_OWN'] = df['person_home_ownership_OWN'].map({False: 0, True: 1})
    df['person_home_ownership_RENT'] = df['person_home_ownership_RENT'].map({False: 0, True: 1})
    df['loan_intent_EDUCATION'] = df['loan_intent_EDUCATION'].map({False: 0, True: 1})
    df['loan_intent_HOMEIMPROVEMENT'] = df['loan_intent_HOMEIMPROVEMENT'].map({False: 0, True: 1})
    df['loan_intent_MEDICAL'] = df['loan_intent_MEDICAL'].map({False: 0, True: 1})
    df['loan_intent_PERSONAL'] = df['loan_intent_PERSONAL'].map({False: 0, True: 1})
    df['loan_intent_VENTURE'] = df['loan_intent_VENTURE'].map({False: 0, True: 1})
    return df

df = arrange_features(df_with_outliers)

def correlation_matrix(df):
    plt.figure(figsize=(15, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

correlation_matrix(df)

X = df.drop(["loan_status"], axis = 1)
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=0),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=50, random_state=42)
}

def model_training(models):
    for model_name, model in models.items():
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        report = classification_report(y_test,y_pred)

        print(f"Accuracy score for {model_name} is: {accuracy}\n")
        print(f"{model_name} Classification Report:\n {report}")
        cm = confusion_matrix(y_test,y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True,fmt='d',xticklabels=['Loan Rejected', 'Loan Approved'], yticklabels=['Loan Rejected', 'Loan Approved'])
        plt.title(f'Confusion Matrix for {model_name}')
        plt.show()

model_training(models)
    


df_trimmed_outliers = pd.read_csv("loan_data.csv", header=0)

def determine_outlier_thresholds_iqr(df, col_name, th1=0.25, th3=0.75):
    quartile1 = df[col_name].quantile(th1)
    quartile3 = df[col_name].quantile(th3)
    iqr = quartile3 - quartile1
    upper_limit = quartile3 + 1.5 * iqr
    lower_limit = quartile1 - 1.5 * iqr
    return lower_limit, upper_limit

def check_outliers_iqr(df, col_name, th1=0.05, th3=0.95):
    lower_limit, upper_limit = determine_outlier_thresholds_iqr(df, col_name, th1, th3)
    mask_upper = df[col_name] > upper_limit
    mask_lower = df[col_name] < lower_limit
    return df[mask_upper | mask_lower]


def remove_outliers_iqr(df, cols, th1=0.05, th3=0.95):
    from tabulate import tabulate
    data = []
    
    for col_name in cols:
        if col_name != 'Outcome':  
            outliers = check_outliers_iqr(df, col_name, th1, th3)
            count = outliers.shape[0]    
            if count > 0:
                df.drop(outliers.index, inplace=True)
            
            remaining_outliers = check_outliers_iqr(df, col_name, th1, th3)
            remaining_outliers_status = not remaining_outliers.empty 
            
            lower_limit, upper_limit = determine_outlier_thresholds_iqr(df, col_name, th1, th3)
            data.append([count > 0, remaining_outliers_status, count, col_name, lower_limit, upper_limit])
    
    table = tabulate(data, headers=['Outliers (Previously)', 'Remaining Outliers', 'Count Removed', 'Column', 'Lower Limit', 'Upper Limit'], tablefmt='rst', numalign='right')
    print("Removing Outliers using IQR")
    print(table)

remove_outliers_iqr(df_trimmed_outliers, ["person_age"])
remove_outliers_iqr(df_trimmed_outliers, ["person_emp_exp"])

df = arrange_features(df_trimmed_outliers)
correlation_matrix(df)

X = df.drop(["loan_status"], axis = 1)
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)


print("##############  Models with trimmed outliers  ##################")
model_training(models)
