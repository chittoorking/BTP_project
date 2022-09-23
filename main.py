import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
import keras
import tensorflow
from keras.layers import Dense
from keras.models import Sequential
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.linear_model import LogisticRegression
st.set_page_config(
    page_title="Digital Twin Manufacturing using Data Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(
    """
    <style>
        .css-hby737, .css-17eq0hr, .css-qbe2hs {
            background-color:    #154360    !important;
            color: black !important;
        }
        div[role="radiogroup"] {
            color:black !important;
            margin-left:8%;
        }
        div[data-baseweb="select"] > div {
            
            color: black;
        }
        div[data-baseweb="base-input"] > div {
            background-color: #aab7b8 !important;
            color: black;
        }
        
        .st-cb, .st-bq, .st-aj, .st-c0{
            color: black !important;
        }
        .st-bs, .st-ez, .st-eq, .st-ep, .st-bd, .st-e2, .st-ea, .st-g9, .st-g8, .st-dh, .st-c0 {
            color: black !important;
        }
        .st-fg, .st-fi {
            background-color: #c6703b !important;
            color: black !important;
        }
        
        .st-g0 {
            border-bottom-color: #c6703b !important;
        }
        .st-fz {
            border-top-color: #c6703b !important;
        }
        .st-fy {
            border-right-color: #c6703b !important;
        }
        .st-fx {
            border-left-color: #c6703b !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown('<h1 style="margin-left:8%; color:#FA8072">Bike Sharing Prediction</h1>', unsafe_allow_html=True)

add_selectbox = st.sidebar.radio(
    "",
    ("About", "Output parameters prediction and Anomaly Prediction", "Team")
)

#def pima():
if add_selectbox == 'About':
    
    st.subheader('ABOUT THE PROJECT')

    st.markdown('<h4>Background</h4>', unsafe_allow_html=True)
    st.markdown('Datasets gathered from robotic manipulator, the Rethink Robotics Baxter (one arm, equipped with a parallel gripper). The dataset contain trajectories generated during the execution of pick and place tasks. The pick and place locations were drawn randomly from two non-overlapping areas with size 50 × 50 cm each, as illustrated in the figure below. The robot was considered to have full executed a task by starting and finishing at the same location. The pick and place locations were randomly chosen.',unsafe_allow_html=True)
    st.markdown('We are predicting the final positions and velocities based on positions , velocities and torques and predicting any anomalies that might could happen',unsafe_allow_html=True)
    
elif add_selectbox == 'Output parameters prediction and Anomaly Prediction':
	
      st.subheader('FINAL PARAMETERS PREDICTION')
      pcaIn= open('pca.pkl','rb')
      pcaLoaded = pickle.load(pcaIn)
#       pickle_in = open('model', 'rb')
#       regressor = pickle.load(pickle_in)
#       anomaly_in=open('clf_model','rb')
#       anomaly_detection=pickle.load(anomaly_in)
      st.markdown("## final values prediction ")
      name = st.text_input("Name:")
      P_j1_t = st.number_input("P_j1_t 0.07 to 0.08", min_value=0.07, max_value=0.08, step=0.00001)
      P_j2_t = st.number_input("P_j2_t -0.05 to -0.04", min_value=-0.05, max_value=-0.04, step=0.00001)
      P_j3_t = st.number_input("P_j3_t -0.17 to -0.16", min_value=-0.17, max_value=-0.16, step=0.00001)
      P_j4_t = st.number_input("P_j4_t -1.67 to -1.60", min_value=-1.67, max_value=-1.60, step=0.00001)
      P_j5_t = st.number_input("P_j5_t 0.1 to 0.11", min_value=0.1, max_value=0.15, step=0.00001)
      P_j6_t = st.number_input("P_j6_t 0.89 to 0.9", min_value=0.89, max_value=0.9, step=0.00001)
      P_j7_t = st.number_input("P_j7_t 0.5 to 0.52", min_value=0.5, max_value=0.52, step=0.00001)
      V_j1_t = st.number_input("V_j1_t -0.0015 to 0.006", min_value=-0.0015, max_value=0.006, step=0.00001)
      V_j2_t = st.number_input("V_j2_t -0.002 to 0.02", min_value=-0.002, max_value=0.02, step=0.00001)
      V_j3_t = st.number_input("V_j3_t 0.004 to 0.2", min_value=0.004, max_value=0.2, step=0.00001)
      V_j4_t = st.number_input("V_j4_t -0.014 to 0.016", min_value=-0.014, max_value=0.016, step=0.00001)
      V_j5_t = st.number_input("V_j5_t -0.008 to 0.017", min_value=-0.008, max_value=0.017, step=0.00001)
      V_j6_t = st.number_input("V_j6_t -0.018 to 0.014", min_value=-0.018, max_value=0.014, step=0.00001)
      V_j7_t = st.number_input("V_j7_t -0.013 to 0.013", min_value=0.001, max_value=0.013, step=0.00001)
      T_j1_t = st.number_input("T_j1_t 1.24 to 1.8", min_value=1.24, max_value=1.8, step=0.00001)
      T_j2_t = st.number_input("T_j2_t -0.73 to -0.46", min_value=-0.73, max_value=-0.46, step=0.00001)
      T_j3_t = st.number_input("T_j3_t -0.31 to -0.1", min_value=-0.31, max_value=-0.1, step=0.00001)
      T_j4_t = st.number_input("T_j4_t -0.28 to -0.1", min_value=-0.28, max_value=-0.1, step=0.00001)
      T_j5_t = st.number_input("T_j5_t -0.025 to 0.129", min_value=-0.025, max_value=0.129, step=0.00001)
      T_j6_t = st.number_input("T_j6_t -1.98 to -1.94", min_value=-1.98, max_value=-1.94, step=0.00001)
      T_j7_t = st.number_input("T_j7_t -0.3 to 0.0", min_value=-0.3, max_value=0.0, step=0.00001)
      pca_df=pd.DataFrame([[P_j1_t,P_j2_t,P_j3_t,P_j4_t,P_j5_t,P_j6_t,P_j7_t,V_j1_t,V_j2_t,V_j3_t,V_j4_t,V_j5_t,V_j6_t,V_j7_t,T_j1_t,T_j2_t,T_j3_t,T_j4_t,T_j5_t,T_j6_t,T_j7_t]],columns=["P_j1_t","P_j2_t","P_j3_t","P_j4_t","P_j5_t","P_j6_t","P_j7_t","V_j1_t","V_j2_t","V_j3_t","V_j4_t","V_j5_t","V_j6_t","V_j7_t","T_j1_t","T_j2_t","T_j3_t","T_j4_t","T_j5_t","T_j6_t","T_j7_t"])
      submit = st.button('Predict')
      if submit:
          principalComponents=pcaLoaded(pca_df)
          st.write('Principal components',principalComponents)
elif add_selectbox == 'Team':
    
    st.subheader('Teamates')

    st.markdown('• <a href="https://www.linkedin.com/in/vamsi-chittoor-331b80189/">Chittoor Vamsi</a>',
                unsafe_allow_html=True)
    st.markdown('• <a href="https://www.linkedin.com/in/keerthi-h-n-182756218/">Keerthi H N </a>',
                unsafe_allow_html=True)

