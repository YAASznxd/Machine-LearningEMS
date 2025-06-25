import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from io import StringIO

def load_data():
    # Load from your preprocessed data
    try:
        data = pd.read_csv("cleaned_hr_dataset.csv")
        features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction']
        return data[features], data['Attrition']
    except:
        # Fallback to original data if preprocessed not available
        data = pd.read_csv("attrition.csv")
        data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
        features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction']
        return data[features], data['Attrition']

def show_page():
    st.header("🌲 Random Forest Prediction")
    st.markdown("""
    **What is Random Forest?**  
    An ensemble method that combines multiple decision trees for better accuracy.
    """)
    
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training section
    st.subheader("Step 1: Train the Model")
    if st.button("Train Model"):
        with st.spinner("Growing forest of decision trees..."):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            st.session_state['rf_model'] = model
            st.session_state['feature_names'] = X.columns.tolist()
            
            # Evaluation
            accuracy = model.score(X_test, y_test)
            st.success(f"Trained! Accuracy: {accuracy:.1%}")
            
            # Feature importance
            st.subheader("Feature Importance")
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots()
            sns.barplot(data=importance, x='Importance', y='Feature', palette='Blues_d')
            ax.set_title("Which Factors Matter Most?")
            st.pyplot(fig)
            
            # Detailed metrics
            st.subheader("Detailed Performance Metrics")
            y_pred = model.predict(X_test)
            st.text(classification_report(y_test, y_pred))
            
            # Confusion matrix
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), 
                        annot=True, fmt='d', 
                        cmap='Blues',
                        ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
    
    # Prediction options
    if 'rf_model' in st.session_state:
        st.subheader("Prediction Options")
        option = st.radio("Choose prediction method:", 
                         ("Single Employee Prediction", "Batch Prediction (Upload CSV)"))
        
        if option == "Single Employee Prediction":
            st.subheader("Step 2: Make Single Prediction")
            st.markdown("Enter employee details to predict attrition risk:")
            
            # Input form
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age", 18, 60, 35)
                income = st.slider("Monthly Income ($)", 1000, 20000, 5000)
            with col2:
                years = st.slider("Years at Company", 0, 30, 5)
                satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            
            if st.button("Predict"):
                model = st.session_state['rf_model']
                prediction = model.predict([[age, income, years, satisfaction]])[0]
                proba = model.predict_proba([[age, income, years, satisfaction]])[0][1]
                
                # Results
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.error(f"🚨 High Risk of Leaving ({proba:.1%} probability)")
                else:
                    st.success(f"✅ Likely to Stay ({(1-proba):.1%} probability)")
                
                # Explanation
                st.subheader("Visual Explanation")
                predictions = [tree.predict([[age, income, years, satisfaction]])[0] 
                             for tree in model.estimators_]
                
                vote_counts = pd.Series(predictions).value_counts()
                vote_counts.index = ['Stay' if i == 0 else 'Leave' for i in vote_counts.index]
                
                fig, ax = plt.subplots()
                vote_counts.plot(kind='bar', color=['green', 'red'], ax=ax)
                ax.set_title(f"How {model.n_estimators} Trees Voted")
                ax.set_ylabel("Number of Trees")
                ax.set_xlabel("Prediction")
                st.pyplot(fig)
        
        else:  # Batch Prediction
            st.subheader("Step 2: Batch Prediction")
            st.markdown("Upload a CSV file with employee data to predict attrition for multiple employees")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    # Read uploaded data
                    df_upload = pd.read_csv(uploaded_file)
                    
                    # Check if required columns exist
                    required_cols = st.session_state['feature_names']
                    missing_cols = [col for col in required_cols if col not in df_upload.columns]
                    
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    else:
                        model = st.session_state['rf_model']
                        
                        # Make predictions
                        X_upload = df_upload[required_cols]
                        predictions = model.predict(X_upload)
                        probas = model.predict_proba(X_upload)[:, 1]
                        
                        # Add predictions to dataframe
                        df_upload['Attrition_Prediction'] = ['Leave' if p == 1 else 'Stay' for p in predictions]
                        df_upload['Attrition_Probability'] = [probas[i] if p == 1 else 1-probas[i] for i, p in enumerate(predictions)]
                        
                        # Show results
                        st.subheader("Prediction Results")
                        st.dataframe(df_upload.style.applymap(
                            lambda x: 'background-color: #ffcccc' if x == 'Leave' else 'background-color: #ccffcc',
                            subset=['Attrition_Prediction']
                        ))
                        
                        # Download results
                        csv = df_upload.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name='attrition_predictions.csv',
                            mime='text/csv'
                        )
                        
                      
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    show_page()