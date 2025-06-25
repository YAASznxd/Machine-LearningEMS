import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_curve, auc)
from io import StringIO

def load_data():
    """Load and prepare the dataset"""
    try:
        # Try loading from preprocessed data first
        data = pd.read_csv("cleaned_hr_dataset.csv")
        features = ['Age', 'MonthlyIncome', 'DistanceFromHome', 
                   'JobSatisfaction', 'WorkLifeBalance']
    except:
        # Fallback to original data if preprocessed not available
        data = pd.read_csv("attrition.csv")
        data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
        features = ['Age', 'MonthlyIncome', 'DistanceFromHome', 
                   'JobSatisfaction', 'WorkLifeBalance']
    
    return data[features], data['Attrition']

def show_page():
    st.header("📊 Logistic Regression Prediction")
    st.markdown("""
    **What is Logistic Regression?**  
    A statistical method for predicting binary outcomes (like Leave/Stay) based on input features.
    """)
    
    # Load and prepare data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Training section
    st.subheader("Step 1: Train the Model")
    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            model = LogisticRegression(max_iter=1000, class_weight='balanced')
            model.fit(X_train, y_train)
            
            # Store model in session state
            st.session_state['lr_model'] = model
            st.session_state['lr_features'] = X.columns.tolist()
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Show results
            st.success("Training complete!")
            st.metric("Model Accuracy", f"{accuracy:.1%}")
            
            # Show coefficients
            st.subheader("Feature Importance (Coefficients)")
            coef_df = pd.DataFrame({
                'Feature': X.columns,
                'Impact': model.coef_[0]
            }).sort_values('Impact', ascending=False)
            
            fig, ax = plt.subplots()
            coef_df.plot.barh(x='Feature', y='Impact', ax=ax, color='skyblue')
            ax.set_title("How Features Affect Attrition Risk")
            ax.set_xlabel("Positive values increase attrition risk")
            st.pyplot(fig)
            
            # Detailed metrics
            st.subheader("Detailed Performance Metrics")
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
            
            # ROC Curve
            st.subheader("ROC Curve")
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
            st.pyplot(fig)
    
    # Prediction options
    if 'lr_model' in st.session_state:
        st.subheader("Prediction Options")
        option = st.radio("Choose prediction method:", 
                         ("Single Employee Prediction", "Batch Prediction (Upload CSV)"))
        
        if option == "Single Employee Prediction":
            st.subheader("Step 2: Make Single Prediction")
            st.markdown("Enter employee details to predict attrition risk:")
            
            # Create input form
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age", 18, 60, 30)
                income = st.slider("Monthly Income ($)", 1000, 20000, 5000)
            with col2:
                distance = st.slider("Distance from Home (miles)", 1, 50, 10)
                satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
                balance = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
            
            if st.button("Predict Attrition"):
                model = st.session_state['lr_model']
                
                # Prepare input
                input_data = [[age, income, distance, satisfaction, balance]]
                
                # Get prediction and probability
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1]
                
                # Show results
                st.subheader("Prediction Result")
                
                if prediction == 1:
                    st.error(f"🚨 High Risk of Leaving ({probability:.1%} probability)")
                    st.markdown("""
                    **Recommended Actions:**
                    - Schedule retention interview
                    - Review compensation package
                    - Consider flexible work options
                    """)
                else:
                    st.success(f"✅ Likely to Stay ({(1-probability):.1%} probability)")
                
                # Explanation
                st.subheader("How This Prediction Works")
                st.markdown(f"""
                The model analyzed these factors:
                - **Age**: {age} years
                - **Monthly Income**: ${income:,}
                - **Commute Distance**: {distance} miles
                - **Job Satisfaction**: {satisfaction}/4
                - **Work-Life Balance**: {balance}/4
                
                Based on patterns learned from historical data, employees with similar profiles 
                have a **{probability:.1%} probability** of leaving the company.
                """)
                
                # Show decision boundary visualization
                st.subheader("Visual Explanation")
                fig, ax = plt.subplots()
                
                # Simplified 2D visualization
                plt.scatter(
                    X['Age'], 
                    X['MonthlyIncome'], 
                    c=y, 
                    alpha=0.3,
                    cmap='coolwarm'
                )
                plt.colorbar(label='Attrition (Red=Leave)')
                plt.scatter(age, income, c='black', s=200, marker='X')
                plt.xlabel("Age")
                plt.ylabel("Monthly Income")
                plt.title("Your Employee in Context (Black X)")
                
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
                    required_cols = st.session_state['lr_features']
                    missing_cols = [col for col in required_cols if col not in df_upload.columns]
                    
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    else:
                        model = st.session_state['lr_model']
                        
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
                        
                        # Show summary stats
                        st.subheader("Prediction Summary")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Employees", len(df_upload))
                            st.metric("Predicted to Stay", f"{sum(predictions == 0)} ({(sum(predictions == 0)/len(predictions)):.1%})")
                        with col2:
                            st.metric("High Risk Employees", f"{sum(predictions == 1)} ({(sum(predictions == 1)/len(predictions)):.1%})")
                            st.metric("Average Risk Score", f"{np.mean(probas):.1%}")
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    show_page()