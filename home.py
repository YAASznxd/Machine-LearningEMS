import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_page():
    # Page configuration
    st.set_page_config(
        layout="wide",
        page_title="HR Analytics Dashboard",
        page_icon="📊"
    )
    
    # Load data
    @st.cache_data
    def load_data():
        data = pd.read_csv("attrition.csv")
        data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
        return data

    data = load_data()

    # ======================
    # INTRODUCTION SECTION
    # ======================
    st.title("📊 HR Analytics Dashboard")
    st.markdown("""
    Welcome to the **Employee Attrition Analysis Platform**. This interactive dashboard helps HR professionals 
    and managers understand workforce turnover patterns and identify key factors influencing employee retention.
    
    **Key Features:**
    - Real-time organizational metrics
    - Visual exploration of attrition drivers
    - Department-level insights
    - Employee satisfaction analysis
    """)
    st.markdown("---")
    
    # ======================
    # KEY METRICS SECTION
    # ======================
    st.header("🔑 Organizational Health Snapshot")
    st.markdown("""
    These high-level metrics provide a quick overview of your workforce composition and attrition patterns.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Employees", len(data))
    with col2:
        st.metric("Attrition Rate", f"{data['Attrition'].mean():.1%}", 
                 help="Percentage of employees who left the company")
    with col3:
        st.metric("Average Age", f"{data['Age'].mean():.1f} years",
                 help="Current average age of employees")
    with col4:
        st.metric("Median Income", f"${data['MonthlyIncome'].median():,.0f}",
                 help="Middle value of monthly salaries")
    
    st.markdown("---")
    
    # ======================
    # VISUAL ANALYSIS SECTION
    # ======================
    st.header("📈 Attrition Drivers Analysis")
    st.markdown("""
    Explore these visualizations to understand what factors contribute most to employee turnover.
    """)
    
    # Visualization 1: Age Distribution
    with st.expander("1. Age vs Attrition Analysis", expanded=True):
        st.markdown("""
        **Insight:** Younger employees tend to leave more frequently than older colleagues.
        This may indicate career growth opportunities or compensation issues for early-career employees.
        """)
        plt.figure(figsize=(10,5))
        sns.histplot(data=data, x='Age', hue='Attrition', bins=20, kde=True, alpha=0.6)
        plt.title("Age Distribution by Attrition Status")
        st.pyplot(plt)
    
    # Visualization 2: Department Breakdown
    with st.expander("2. Department-wise Attrition"):
        st.markdown("""
        **Insight:** Compare attrition rates across different departments.
        High turnover in specific departments may indicate management or role-specific issues.
        """)
        plt.figure(figsize=(10,5))
        sns.countplot(data=data, x='Department', hue='Attrition', palette='Blues')
        plt.title("Attrition Counts by Department")
        st.pyplot(plt)
    
    # Visualization 3: Job Satisfaction
    with st.expander("3. Job Satisfaction Analysis"):
        st.markdown("""
        **Insight:** Employees who leave often report lower job satisfaction.
        Satisfaction levels vary significantly by department.
        """)
        plt.figure(figsize=(10,5))
        sns.boxplot(data=data, x='JobSatisfaction', y='Department', hue='Attrition')
        plt.title("Job Satisfaction Distribution by Department")
        st.pyplot(plt)
    
    # Visualization 4: Income Analysis
    with st.expander("4. Income vs Attrition"):
        st.markdown("""
        **Insight:** Employees who leave tend to have lower monthly incomes.
        Compensation is a significant factor in retention.
        """)
        plt.figure(figsize=(10,5))
        sns.violinplot(data=data, x='Attrition', y='MonthlyIncome', palette='Greens')
        plt.title("Income Distribution by Attrition Status")
        st.pyplot(plt)
    
    # Visualization 5: Overtime Impact
    with st.expander("5. Overtime Impact Analysis"):
        st.markdown("""
        **Insight:** Employees working overtime are significantly more likely to leave.
        Consider workload balance and overtime compensation policies.
        """)
        plt.figure(figsize=(8,5))
        data.groupby('OverTime')['Attrition'].mean().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title("Attrition Rate by Overtime Status")
        plt.ylabel("Attrition Rate")
        st.pyplot(plt)
    
    # Visualization 6: Correlation Heatmap
    with st.expander("6. Feature Relationships"):
        st.markdown("""
        **Insight:** Explore how different factors relate to each other.
        Strong correlations may reveal hidden patterns in your workforce data.
        """)
        plt.figure(figsize=(12,8))
        numeric_data = data.select_dtypes(include=np.number)
        corr = numeric_data.corr()
        sns.heatmap(corr, annot=True, fmt=".1f", cmap='coolwarm', center=0)
        plt.title("Correlation Between Numerical Features")
        st.pyplot(plt)
    
    # ======================
    # CONCLUSION SECTION
    # ======================
    st.markdown("---")
    st.header("💡 Key Takeaways")
    st.markdown("""
    1. **Younger employees** and those with **lower incomes** are most at risk of leaving
    2. **Overtime work** significantly increases attrition likelihood
    3. **Job satisfaction** varies by department and impacts retention
    4. Certain **departments** may need targeted retention strategies
    
    Use these insights to develop data-driven HR policies and retention programs.
    """)

if __name__ == "__main__":
    show_page()