import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.diabetes import MissingValuesHandler

# carrega o dataset
filename = 'diabetes.csv'
diabetes_df = pd.read_csv(filename)

# separa o vetor de classes
y = diabetes_df['Outcome']
X = diabetes_df.drop('Outcome', axis=1)

# divide o conjunto de teste e treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# algumas colunas com valor 0 na verdade sao missing values - transformar para NaN e trata-los
missing_values_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
X_train, X_test = MissingValuesHandler.set_numpy_missing_values(X_train, X_test, 0, missing_values_columns)
X_train, X_test = MissingValuesHandler.imputation_with_mean(X_train, X_test, missing_values_columns)

# salvar os datasets
training_df = pd.concat((X_train, y_train), axis=1)
test_df = pd.concat((X_test, y_test), axis=1)

training_df.to_csv('hepatitis_training.csv', index=False)
test_df.to_csv('hepatitis_test.csv', index=False)


