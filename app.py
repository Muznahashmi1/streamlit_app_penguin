import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#make containers
header = st.container()
datasets= st.container()
feature = st.container()
model_training = st.container()

with header:
    st.title('Streamlit App')
    st.text('This is a streamlit app')

with datasets:
    st.header('Datasets')
    st.text('Choose a dataset')

    # import dataset
    df= sns.load_dataset('penguins')
    df=df.dropna()
    st.write(df.head(10))
    #plot graph
    st.subheader('Species of Penguins')
    st.bar_chart(df['species'].value_counts())


    #other plots
    st.subheader('Body Mass of Penguins')
    st.bar_chart(df['body_mass_g'].sample(30))

with feature:
    st.header('Features')
    st.text('Choose a feature')
    st.markdown("***Feature:01*** This will tell about Body Mass of Penguins")
    st.markdown("***Feature:02*** This will tell about Bill Length of Penguins")
    st.markdown("***Feature:03*** This will tell about Bill Depth of Penguins")
    st.markdown("***Feature:04*** This will tell about Flipper Size of Penguins")


with model_training:
    st.header('Model Training')
    st.text('Train a model')
    # making columns
    input, display= st.columns(2)
    #phle columns mai selection points hon
    max_depth= input.slider('Body Mass of Penguin', min_value=10, max_value=100, value=20, step=5)

n_estimators = input.selectbox ("How many trees in the Random Forest?", options=[10, 50, 100, 200, 'No limit' ])
    
#input list of features

input.write(df.columns)
    
    #input feature from user

input_feature= input.text_input('which feature do you want to use?')


#machine learning

model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)

# apply a condition

if n_estimators == 'No limit':
    model = RandomForestClassifier(max_depth=max_depth, random_state=0)
else:
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)

#define X and y
X= df[[input_feature]]
y= df[['body_mass_g']]

#fit the model
model.fit(X,y)
predict= model.predict(y)

#Display metrics
display.subheader('Mean Absolute Error of the model is')
display.write(mean_absolute_error(y, predict))
display.subheader('mean squared error of the model is')
display.write(mean_squared_error(y, predict))
display.subheader('r2 score of the model is')
display.write(r2_score(y, predict))



