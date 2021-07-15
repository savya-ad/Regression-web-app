import streamlit as st
import pandas as pd
import base64
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# predict 
def pred(df_train, df_test):
    X_train=df_train.iloc[:,:-1].values
    Y_train=df_train.iloc[:,-1].values
    # x_train,x_test,y_train,y_test = train_test_split( X, Y, test_size=0.2,random_state=0)
    X_test= df_test.iloc[:,:-1]
    Y_test =df_test.iloc[:,-1]
    
    model=RandomForestRegressor()
    model.fit(X_train, Y_train)
    pred=model.predict(X_test)

    return Y_test, pred

# Calculates performance metrics
def calc_metrics(Y_actual,Y_predicted ):
    mse=mean_squared_error(Y_actual,Y_predicted)
    rmse=np.sqrt(mse)
    r_squared=r2_score(Y_actual,Y_predicted)

    mse_series= pd.Series(mse,name='MSE')
    rmse_series= pd.Series(rmse,name='RMSE')
    r2score_series= pd.Series(r_squared,name='R_squared')

    
    df = pd.concat( [mse_series, rmse_series, r2score_series], axis=1 )
    return df

# Load example data
def load_example_data():
    df1 = pd.read_csv('vegitation_data_train.csv')
    df2=pd.read_csv('vegitation_data_test.csv')
    return df1,df2

# Download performance metrics
def filedownload(df,name):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={name}.csv>Download CSV File</a>'
    return href

# Sidebar - Header
st.sidebar.header('Input panel')
st.sidebar.markdown("""
[Example CSV file](https://raw.githubusercontent.com/dataprofessor/model_performance_app/main/Y_example.csv)
""")

# Sidebar panel - Upload input file
uploaded_file = st.sidebar.file_uploader('Upload your train CSV file', type=['csv'])
uploaded_test = st.sidebar.file_uploader('Upload your test CSV file', type=['csv'])

# Sidebar panel - Performance metrics
performance_metrics = ['MSE', 'RMSE', 'R_squared']
selected_metrics = st.sidebar.multiselect('Performance metrics', performance_metrics, performance_metrics)

# Main panel
image = Image.open('logo.png')
st.image(image, width = 500)
st.title('Model Performance Calculator App')
st.markdown("""
This app calculates the model performance metrics given the actual and predicted values.
* **Python libraries:** `base64`, `pandas`, `streamlit`, `scikit-learn`
""")

if (uploaded_file and uploaded_test) is not None:
    train_df = pd.read_csv(uploaded_file)
    test_df = pd.read_csv(uploaded_test)

    y_actual, y_pred = pred(train_df, test_df)

    metrics_df = calc_metrics(y_actual,y_pred)

    y_actual_series =pd.Series(y_actual,name="Actual")
    y_predicted_series =pd.Series(y_pred,name="Predicted")
    pred_data=pd.concat([y_actual_series, y_predicted_series],axis=1)

    selected_metrics_df = metrics_df[ selected_metrics ]
    st.header('train data')
    st.write(train_df)
    st.header('test data')
    st.write(test_df)

    st.header('Predicted data')
    st.write(pred_data)
    st.markdown(filedownload(pred_data,'predicted_data'), unsafe_allow_html=True)

    st.header('Performance metrics')
    st.write(selected_metrics_df)
    st.markdown(filedownload(selected_metrics_df,'performance_metrics'), unsafe_allow_html=True)
else:
    st.info('Awaiting the upload of the input file.')
    if st.button('Use Example Data'):
        train_df, test_df = load_example_data()
        y_actual, y_pred = pred(train_df, test_df)

        metrics_df = calc_metrics(y_actual,y_pred)

        y_actual_series =pd.Series(y_actual,name="Actual")
        y_predicted_series =pd.Series(y_pred,name="Predicted")
        pred_data=pd.concat([y_actual_series, y_predicted_series],axis=1)

        selected_metrics_df = metrics_df[ selected_metrics ]
        st.header('train data')
        st.write(train_df)
        st.header('test data')
        st.write(test_df)

        st.header('Predicted data')
        st.write(pred_data)
        st.markdown(filedownload(pred_data,'predicted_data'), unsafe_allow_html=True)

        st.header('Performance metrics')
        st.write(selected_metrics_df)
        st.markdown(filedownload(selected_metrics_df,'performance_metrics'), unsafe_allow_html=True)
