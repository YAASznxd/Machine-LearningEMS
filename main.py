import streamlit as st

st.set_page_config(
    page_title="Employee Attrition Predictor",
    layout="wide",
    page_icon="👨‍💼"
)
st.sidebar.image("logo.png", width=200) 


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home Dashboard",
    "Logistic Regression",
    "Decision Tree", 
    "Random Forest",
    "K-Nearest Neighbors",
    "Employee Engagement Analysis",
    "Employee Attrition Prediction"
])

# Main content router
if page == "Home Dashboard":
    from home import show_page
    show_page()
elif page=="Employee Engagement Analysis":
    from engagement_analysis import show_page
    show_page()
elif page=="Employee Attrition Prediction":
    from attrition_prediction import show_page
    show_page()
elif page == "Logistic Regression":
    from Logistic_Regression import show_page
    show_page()
elif page == "Decision Tree":
    from Decision_Tree import show_page
    show_page()
elif page == "Random Forest":
    from Random_Forest import show_page
    show_page()
elif page == "K-Nearest Neighbors":
    from K_Nearest import show_page
    show_page()

