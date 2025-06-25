import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_curve, auc)
from io import StringIO

def load_data():
    """Load and prepare the dataset"""
    try:
        # Try loading from preprocessed data first
        data = pd.read_csv("cleaned_hr_dataset.csv")
        features = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance']
    except:
        # Fallback to original data if preprocessed not available
        data = pd.read_csv("attrition.csv")
        data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
        features = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance']
    
    return data[features], data['Attrition']

def show_page():
    st.header("🌳 Decision Tree Prediction")
    st.markdown("""
    **What is a Decision Tree?**  
    A flowchart-like model that makes decisions by splitting data on feature values.
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
    
    # Parameter tuning
    with st.expander("Tree Configuration"):
        max_depth = st.slider("Max tree depth", 1, 10, 3)
        min_samples_split = st.slider("Min samples to split", 2, 20, 2)
        criterion = st.radio("Split criterion", ['gini', 'entropy'])
    
    if st.button("Train Model"):
        with st.spinner(f"Building decision tree (max depth={max_depth})..."):
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion=criterion,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Store model in session state
            st.session_state['dt_model'] = model
            st.session_state['dt_features'] = X.columns.tolist()
            
            # Evaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Show results
            st.success(f"Trained! Accuracy: {accuracy:.1%}")
            
            # Visualize tree
            st.subheader("Decision Tree Structure")
            fig, ax = plt.subplots(figsize=(20,10))
            plot_tree(model, 
                     feature_names=X.columns,
                     class_names=['Stay', 'Leave'],
                     filled=True,
                     rounded=True,
                     ax=ax)
            st.pyplot(fig)
            
            # Show text representation
            st.subheader("Tree Rules")
            tree_rules = export_text(model, feature_names=list(X.columns))
            st.text(tree_rules)
            
            # Feature importance
            st.subheader("Feature Importance")
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots()
            importance.plot.barh(x='Feature', y='Importance', ax=ax, color='green')
            ax.set_title("Which Factors Matter Most?")
            st.pyplot(fig)
            
            # Detailed metrics
            st.subheader("Detailed Performance Metrics")
            st.text(classification_report(y_test, y_pred))
            
            # Confusion matrix
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), 
                        annot=True, fmt='d', 
                        cmap='Greens',
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
            ax.plot(fpr, tpr, color='darkgreen', lw=2, 
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
    if 'dt_model' in st.session_state:
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
                satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
                balance = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
            
            if st.button("Predict"):
                model = st.session_state['dt_model']
                prediction = model.predict([[age, income, satisfaction, balance]])[0]
                proba = model.predict_proba([[age, income, satisfaction, balance]])[0][1]
                
                # Results
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.error(f"🚨 High Risk of Leaving ({proba:.1%} probability)")
                else:
                    st.success(f"✅ Likely to Stay ({(1-proba):.1%} probability)")
                
                # Explanation
                st.subheader("Visual Explanation")
                
                # Decision Path Visualization
                st.markdown("**Decision Path**")
                fig, ax = plt.subplots(figsize=(15,8))
                plot_tree(model, 
                         feature_names=X.columns,
                         class_names=['Stay', 'Leave'],
                         filled=True,
                         max_depth=2,  # Show first 2 levels for clarity
                         ax=ax)
                st.pyplot(fig)
                
                # Feature Space Visualization
                st.markdown("**Feature Space**")
                fig, ax = plt.subplots()
                scatter = ax.scatter(X['Age'], X['MonthlyIncome'], c=y, cmap='coolwarm', alpha=0.5)
                ax.scatter(age, income, c='black', s=200, marker='X')
                ax.set_xlabel("Age")
                ax.set_ylabel("Monthly Income")
                ax.legend(*scatter.legend_elements(), title="Attrition")
                ax.set_title("Your Employee in Context (Black X)")
                st.pyplot(fig)
                
                # Decision Explanation
                st.markdown("""
                **How the Decision Was Made:**
                1. First checked if {} > {}
                2. Then examined {} > {}
                3. Final decision based on leaf node purity
                """.format(
                    "Age" if age > 30 else "Income",
                    "30" if age > 30 else "5000",
                    "Job Satisfaction" if satisfaction < 3 else "Work-Life Balance",
                    "3" if satisfaction < 3 else "3"
                ))
        
        else:  # Batch Prediction
            st.subheader("Step 2: Batch Prediction")
            st.markdown("Upload a CSV file with employee data to predict attrition for multiple employees")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    # Read uploaded data
                    df_upload = pd.read_csv(uploaded_file)
                    
                    # Check if required columns exist
                    required_cols = st.session_state['dt_features']
                    missing_cols = [col for col in required_cols if col not in df_upload.columns]
                    
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    else:
                        model = st.session_state['dt_model']
                        
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
                            file_name='dt_attrition_predictions.csv',
                            mime='text/csv'
                        )
                        
                 
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    show_page()