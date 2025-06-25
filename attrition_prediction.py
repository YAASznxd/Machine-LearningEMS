import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data():
    df = pd.read_csv("cleaned_hr_dataset.csv")
    df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})
    features = ['Age', 'MonthlyIncome', 'JobSatisfaction', 
               'WorkLifeBalance', 'YearsAtCompany']
    return df[features], df['Attrition']

def show_page():
    st.title("Employee Attrition Prediction")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model selection
    model_type = st.selectbox("Select Algorithm", 
                            ["Logistic Regression", "Decision Tree", 
                             "Random Forest", "K-Nearest Neighbors"])
    
    # Model training
    if st.button("Train Model"):
        with st.spinner(f"Training {model_type}..."):
            if model_type == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=1000)
            elif model_type == "Decision Tree":
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier(max_depth=3)
            elif model_type == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100)
            else:
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier(n_neighbors=5)
                
            model.fit(X_train, y_train)
            st.session_state.model = model
            st.session_state.model_type = model_type
            st.session_state.features = X.columns.tolist()
            
            # Evaluation
            y_pred = model.predict(X_test)
            accuracy = model.score(X_test, y_test)
            report = classification_report(y_test, y_pred)
            
            st.success(f"{model_type} trained! Accuracy: {accuracy:.1%}")
            st.text(report)
    
    # Prediction options
    if 'model' in st.session_state:
        prediction_option = st.radio("Prediction Method:", 
                                   ["Single Employee", "Batch Prediction"])
        
        if prediction_option == "Single Employee":
            st.subheader("Single Employee Prediction")
            inputs = {}
            cols = st.columns(2)
            with cols[0]:
                inputs['Age'] = st.slider("Age", 18, 65, 30)
                inputs['MonthlyIncome'] = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
            with cols[1]:
                inputs['JobSatisfaction'] = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
                inputs['WorkLifeBalance'] = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
                inputs['YearsAtCompany'] = st.slider("Years at Company", 0, 40, 3)
            
            if st.button("Predict"):
                model = st.session_state.model
                prediction = model.predict([list(inputs.values())])[0]
                proba = model.predict_proba([list(inputs.values())])[0][1]
                
                if prediction == 1:
                    st.error(f"High attrition risk ({proba:.1%} probability)")
                else:
                    st.success(f"Low attrition risk ({(1-proba):.1%} probability)")
        
        else:  # Batch Prediction
            st.subheader("Batch Prediction")
            uploaded_file = st.file_uploader("Upload CSV file with employee data", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    
                    # Check required columns
                    required_cols = st.session_state.features
                    missing_cols = [col for col in required_cols if col not in df_upload.columns]
                    
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    else:
                        model = st.session_state.model
                        X_upload = df_upload[required_cols]
                        
                        # Make predictions
                        predictions = model.predict(X_upload)
                        probas = model.predict_proba(X_upload)[:, 1]
                        
                        # Add predictions to dataframe
                        df_upload['Attrition_Prediction'] = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
                        df_upload['Attrition_Probability'] = probas
                        
                        # Show results
                        st.subheader("Prediction Results")
                        st.dataframe(df_upload)
                        
                        # Download results
                        csv = df_upload.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name='attrition_predictions.csv',
                            mime='text/csv'
                        )
                        
                        # Show summary
                        st.write(f"High Risk Employees: {sum(predictions == 1)}")
                        st.write(f"Low Risk Employees: {sum(predictions == 0)}")
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    show_page()