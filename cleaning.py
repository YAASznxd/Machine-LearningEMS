import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, f_classif

# Load the raw dataset
df = pd.read_csv("data/attrition.csv")

# 1. Drop uninformative or identifier columns
columns_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
df.drop(columns=columns_to_drop, axis=1, inplace=True)

# 2. Handle missing values
print("Missing values before handling:")
print(df.isnull().sum())

# Fill missing values - example for numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Fill numerical missing with median
for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)
    
# Fill categorical missing with mode
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())

# 3. Encode categorical features
label_encoder = LabelEncoder()
df['Attrition'] = label_encoder.fit_transform(df['Attrition'])

# One-hot encode other categorical features
df = pd.get_dummies(df, columns=[col for col in categorical_cols if col != 'Attrition'], 
                    drop_first=True)

# 4. Split data into features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# 5. Feature selection using ANOVA F-value
selector = SelectKBest(score_func=f_classif, k=15)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask]
print("\nSelected Features:")
print(selected_features)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)



# 8. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save processed data
np.savez("Assets/processed_data.npz", 
         X_train=X_train, X_test=X_test, 
         y_train=y_train, y_test=y_test)



# Save cleaned dataset for future use
df.to_csv("Assets/cleaned_hr_dataset.csv", index=False)

print("\nData preprocessing completed successfully!")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Class distribution (training): {pd.Series(y_train).value_counts(normalize=True)}")