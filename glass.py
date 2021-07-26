# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


features_list=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
@st.cache()
def prediction(model,RI, Na, Mg, Al, Si, K, Ca, Ba, Fe):
  pred=model.predict([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])
  if pred[0]==1:
    return 'building windows float processed'
  elif pred[0]==2:
    return 'building windows non float processed'
  elif pred[0]==3:
    return 'vehicle windows float processed'
  elif pred[0]==4:
    return 'vehicle windows non float processed'
  elif pred[0]==5:
    return 'containers'
  elif pred[0]==6:
    return 'tableware'
  else:
    return 'headlamp'

st.title("GLASSES USED")
st.sidebar.title('CHOOSE THE PARAMETERS')

if st.sidebar.checkbox('Show raw data'):
  st.subheader('Glass Type Data set')
  st.dataframe(glass_df)

st.sidebar.subheader('Visualisation selector')
plot_list=st.sidebar.multiselect("select the visualiser", ('Correlation Heatmap', 'Line Chart', 'Area Chart', 'Count Plot','Pie Chart', 'Box Plot'))

if 'Line Chart' in plot_list:
  st.subheader("Line chart")
  st.line_chart(glass_df) 
# Display area chart    
if 'Area Chart' in plot_list:
  st.subheader("Area chart")
  st.area_chart(glass_df)

import seaborn as sns 
import matplotlib.pyplot as plt
# Display correlation heatmap using seaborn module and 'st.pyplot()'
st.set_option('deprecation.showPyplotGlobalUse', False)
# Display count plot using seaborn module and 'st.pyplot()' 
if 'Correlation Heatmap' in plot_list:
  st.subheader("Correlation heatmap")
  plt.figure(figsize=(8,4))
  sns.heatmap(glass_df.corr(),annot=True)
  st.pyplot()
# Display pie chart using matplotlib module and 'st.pyplot()'   
if 'Count Plot' in plot_list:
  st.subheader("Cout plot")
  plt.figure(figsize=(8,4))
  sns.countplot(glass_df['GlassType'])
  st.pyplot()

if 'Pie Chart' in plot_list:
  st.subheader("Pie chart")
  plt.figure(figsize=(8,4))
  count=glass_df['GlassType'].value_counts()
  plt.pie(count,labels=count.index,autopct='%1.2f%%')
  st.pyplot()

if "Box Plot" in plot_list:
  st.subheader('Box plot')
  variable=st.sidebar.selectbox('Choose the variable',list(glass_df.columns))

  plt.figure(figsize=(8,4))
  sns.boxplot(glass_df[variable])
  st.pyplot()


st.sidebar.subheader('Select the features')
RI=st.sidebar.slider('RI',float(glass_df['RI'].min()),float(glass_df['RI'].max()))
Na=st.sidebar.slider('Na',float(glass_df['Na'].min()),float(glass_df['Na'].max()))
Mg=st.sidebar.slider('Mg',float(glass_df['Mg'].min()),float(glass_df['Mg'].max()))
Al=st.sidebar.slider('Al',float(glass_df['Al'].min()),float(glass_df['Al'].max()))
Si=st.sidebar.slider('Si',float(glass_df['Si'].min()),float(glass_df['Si'].max()))
K=st.sidebar.slider('K',float(glass_df['K'].min()),float(glass_df['K'].max()))
Ca=st.sidebar.slider('Ca',float(glass_df['Ca'].min()),float(glass_df['Ca'].max()))
Ba=st.sidebar.slider('Ba',float(glass_df['Ba'].min()),float(glass_df['Ba'].max()))
Fe=st.sidebar.slider('Fe',float(glass_df['Fe'].min()),float(glass_df['Fe'].max()))

st.sidebar.subheader('Select classifier')
clf=st.sidebar.selectbox('classifier',('RandomForestClassifier','SVC','LogisticRegression'))

from sklearn.metrics import plot_confusion_matrix
if clf=='SVC':
    st.sidebar.subheader('Hyper parameters tuning')
    c=st.sidebar.number_input('C(Error rate)',1,100,step=1)
    g=st.sidebar.number_input('Gamma',1,100,step=1)
    k=st.sidebar.radio('Kernel',('linear','rbf','poly'))

    if st.sidebar.button('Classifiy'):
        st.subheader('SVC')
        svc=SVC(kernel=k,C=c,gamma=g)
        svc.fit(X_train,y_train)
        score=svc.score(X_test,y_test)
        glass_type=prediction(svc,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe)
        st.write("score is ",score,' and the predicted value is ',glass_type)
        plot_confusion_matrix(svc,X_test,y_test)
        st.pyplot()
elif clf=='RandomForestClassifier':
    st.sidebar.subheader('Hyper parameters tuning')
    est=st.sidebar.number_input('n estimators',100,5000,step=20)
    dep=st.sidebar.number_input('maximum depth',1,100,step=1)

    if st.sidebar.button('Classifiy'):
        st.subheader('RandomForestClassifier')
        rf_clf=RandomForestClassifier(n_estimators=est,max_depth=dep,n_jobs=-1)
        rf_clf.fit(X_train,y_train)
        score=rf_clf.score(X_test,y_test)
        glass_type=prediction(rf_clf,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe)
        st.write("score is ",score,' and the predicted value is ',glass_type)
        plot_confusion_matrix(rf_clf,X_test,y_test)
        st.pyplot()
elif clf=='LogisticRegression':
    st.sidebar.subheader('Hyper parameters tuning')
    c=st.sidebar.number_input('C(Error rate)',1,100,step=1)
    m=st.sidebar.number_input("Max Iterations",10,5000,step=10)

    if st.sidebar.button("Classify"):
        st.subheader('LogisticRegression')
        lr=LogisticRegression(C=c,max_iter=m)
        lr.fit(X_train,y_train)
        score=lr.score(X_test,y_test)
        glass_type=prediction(lr,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe)
        st.write("score is ",score,' and the predicted value is ',glass_type)
        plot_confusion_matrix(lr,X_test,y_test)
        st.pyplot()
