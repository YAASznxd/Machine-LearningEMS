import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    return pd.read_csv("attrition.csv")

def show_page():
    st.title("Employee Engagement Analysis")
    df = load_data()
    
    # Key Metrics Section
    st.header("Key Engagement Metrics")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Avg Job Satisfaction", f"{df['JobSatisfaction'].mean():.1f}/4")
    with cols[1]:
        st.metric("Avg Work-Life Balance", f"{df['WorkLifeBalance'].mean():.1f}/4")
    with cols[2]:
        st.metric("Overtime Rate", f"{df['OverTime'].eq('Yes').mean():.1%}")
    
    # Visualizations Section
    st.header("Engagement Insights")
    
    # 1. Simple Bar Chart - Satisfaction by Department
    st.subheader("Job Satisfaction by Department")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=df, x='Department', y='JobSatisfaction', ci=None, palette='Blues')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # 2. Work-Life Balance Distribution
    st.subheader("Work-Life Balance Distribution")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.countplot(data=df, x='WorkLifeBalance', palette='Greens')
    ax.set_xlabel("Work-Life Balance Rating (1-4)")
    st.pyplot(fig)
    
    # 3. Satisfaction vs Years at Company
    st.subheader("Satisfaction Over Time")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=df, x='YearsAtCompany', y='JobSatisfaction', ci=None, color='purple')
    ax.set_xlabel("Years at Company")
    ax.set_ylabel("Avg Job Satisfaction")
    st.pyplot(fig)
    
    # 4. Simple Attrition Comparison
    st.subheader("Engagement by Attrition Risk")
    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    sns.boxplot(data=df, x='Attrition', y='JobSatisfaction', ax=ax[0])
    sns.boxplot(data=df, x='Attrition', y='WorkLifeBalance', ax=ax[1])
    ax[0].set_title("Job Satisfaction")
    ax[1].set_title("Work-Life Balance")
    st.pyplot(fig)
    
    # 5. Overtime Impact
    st.subheader("Overtime Impact on Engagement")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=df, x='OverTime', y='JobSatisfaction', ci=None, palette='Oranges')
    ax.set_xlabel("Works Overtime")
    st.pyplot(fig)

if __name__ == "__main__":
    show_page()