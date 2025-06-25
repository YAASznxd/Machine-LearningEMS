import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_curve, auc)
from io import StringIO

def load_data():
    """Load and prepare the dataset"""
    try:
        # Try loading from preprocessed data first
        data = pd.read_csv("cleaned_hr_dataset.csv")
        features = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'JobSatisfaction']
    except:
        # Fallback to original data if preprocessed not available
        data = pd.read_csv("attrition.csv")
        data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
        features = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'JobSatisfaction']
    
    return data[features], data['Attrition']

def show_page():
    st.header("📏 K-Nearest Neighbors Prediction")
    st.markdown("""
    **What is KNN?**  
    An instance-based learning algorithm that finds similar employees in historical data to predict outcomes.
    """)
    
    # Load and prepare data
    X, y = load_data()
    
    # Scale data for KNN
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Training section
    st.subheader("Step 1: Train the Model")
    
    # Parameter tuning
    with st.expander("Algorithm Settings"):
        n_neighbors = st.slider("Number of neighbors (k)", 1, 20, 5)
        weights = st.radio("Weighting method", ['uniform', 'distance'])
        algorithm = st.selectbox("Algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'])
    
    if st.button("Train Model"):
        with st.spinner(f"Finding {n_neighbors} optimal neighbors..."):
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm
            )
            model.fit(X_train, y_train)
            
            # Store model in session state
            st.session_state['knn_model'] = model
            st.session_state['knn_scaler'] = scaler
            st.session_state['knn_features'] = X.columns.tolist()
            
            # Evaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Show results
            st.success(f"Trained with {n_neighbors} neighbors! Accuracy: {accuracy:.1%}")
            
            # Feature distributions
            st.subheader("Feature Distributions by Attrition")
            fig, ax = plt.subplots(2, 2, figsize=(12,8))
            for i, feature in enumerate(X.columns):
                sns.boxplot(x=y, y=X[feature], ax=ax[i//2,i%2])
                ax[i//2,i%2].set_title(feature)
                ax[i//2,i%2].set_xticklabels(['Stay', 'Leave'])
            plt.tight_layout()
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
    if 'knn_model' in st.session_state:
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
                distance = st.slider("Distance from Home", 1, 50, 10)
                satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            
            if st.button("Predict"):
                model = st.session_state['knn_model']
                scaler = st.session_state['knn_scaler']
                
                # Scale inputs
                input_scaled = scaler.transform([[age, income, distance, satisfaction]])
                
                # Get prediction and neighbors
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0][1]
                distances, indices = model.kneighbors(input_scaled)
                
                # Get neighbor details
                neighbors = X.iloc[indices[0]].copy()
                neighbors['Attrition'] = y.iloc[indices[0]].values
                neighbors['Distance'] = distances[0]
                
                # Results
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.error(f"🚨 High Risk of Leaving ({proba:.1%} probability)")
                else:
                    st.success(f"✅ Likely to Stay ({(1-proba):.1%} probability)")
                
                # Explanation
                st.subheader("Visual Explanation")
                
                # Neighbor visualization
                st.markdown("**Most Similar Employees**")
                fig, ax = plt.subplots(figsize=(10,6))
                scatter = ax.scatter(
                    neighbors['Age'], 
                    neighbors['MonthlyIncome'],
                    c=neighbors['Attrition'],
                    cmap='coolwarm',
                    s=100,
                    alpha=0.7
                )
                ax.scatter(age, income, c='black', s=200, marker='X')
                ax.set_xlabel("Age")
                ax.set_ylabel("Monthly Income")
                ax.legend(*scatter.legend_elements(), title="Attrition")
                ax.set_title("Your Employee (X) Among Similar Cases")
                st.pyplot(fig)
                
                # Neighbor details
                st.markdown("**Similar Employee Details**")
                st.dataframe(neighbors.style.format({
                    'MonthlyIncome': '${:,.0f}',
                    'Distance': '{:.2f}',
                    'JobSatisfaction': '{:.0f}/4'
                }).background_gradient(subset=['Distance'], cmap='Blues_r'))
                
                # Explanation text
                st.markdown(f"""
                **How the Prediction Was Made:**
                - Found {model.n_neighbors} most similar employees
                - {sum(neighbors['Attrition'])} left the company
                - {len(neighbors)-sum(neighbors['Attrition'])} stayed with the company
                - Average distance to neighbors: {np.mean(distances):.2f}
                """)
        
        else:  # Batch Prediction
            st.subheader("Step 2: Batch Prediction")
            st.markdown("Upload a CSV file with employee data to predict attrition for multiple employees")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    # Read uploaded data
                    df_upload = pd.read_csv(uploaded_file)
                    
                    # Check if required columns exist
                    required_cols = st.session_state['knn_features']
                    missing_cols = [col for col in required_cols if col not in df_upload.columns]
                    
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    else:
                        model = st.session_state['knn_model']
                        scaler = st.session_state['knn_scaler']
                        
                        # Scale and predict
                        X_upload = df_upload[required_cols]
                        X_upload_scaled = scaler.transform(X_upload)
                        predictions = model.predict(X_upload_scaled)
                        probas = model.predict_proba(X_upload_scaled)[:, 1]
                        
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
                            file_name='knn_attrition_predictions.csv',
                            mime='text/csv'
                        )
                        
                     
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    show_page()