# from crypt import methods
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO

import base64

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/products')
def services():
    return render_template("products.html")

@app.route('/stock')
def stock1():
    return render_template("stock.html")


@app.route('/stock', methods=['GET','POST'])
def stock():
    if request.method == 'POST':
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]
        stock_name = request.form["stock_name"]


        
        user_input = stock_name
        start = start_date
        end = end_date
                
        df = data.DataReader(user_input,'yahoo', start, end)

        dff = df.reset_index()
        dff.head()

        len(dff)

        opn = dff[['Open']]

        ds = opn.values
        plt.plot(ds)
        plt.title('Opening price')

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        openImg = base64.b64encode(img.getvalue()).decode('utf8')


       # this graph as output
       # cls = dff[['Close']]
       # cl = cls.values
        plt.plot(dff.Close)
        plt.title('Closing price')

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        closeImg = base64.b64encode(img.getvalue()).decode('utf8')


        #Using MinMaxScaler for normalizing data between 0 & 1
        scaler = MinMaxScaler(feature_range=(0,1))
        ds_scaled = scaler.fit_transform(np.array(ds).reshape(-1,1))

        len(ds_scaled), len(ds)

        #Defining test and train data sizes
        train_size = int(len(ds_scaled)*0.70)
        test_size = len(ds_scaled) - train_size

        train_size,test_size

        #Splitting data between train and test
        ds_train = ds_scaled[0:train_size,:]
        ds_test = ds_scaled[train_size:len(ds_scaled),:1]

        len(ds_train),len(ds_test)

        #creating dataset in time series for LSTM model 
        #X[100,120,140,160,180] : Y[200]
        def create_ds(dataset,step):
            Xtrain, Ytrain = [], []
            for i in range(len(dataset)-step-1):
                a = dataset[i:(i+step), 0]
                Xtrain.append(a)
                Ytrain.append(dataset[i + step, 0])
            return np.array(Xtrain), np.array(Ytrain)

        #Taking 100 days price as one record for training
        time_stamp = 100
        x_train, y_train = create_ds(ds_train,time_stamp)
        x_test, y_test = create_ds(ds_test,time_stamp)

        x_train.shape,y_train.shape

        x_test.shape, y_test.shape

       #Reshaping data to fit into LSTM model
        x_train = x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
        x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)

        from keras.models import Sequential
        from keras.layers import Dense, LSTM

        #Creating LSTM model using keras
        model = Sequential()
        model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
        model.add(LSTM(units=50,return_sequences=True))
        model.add(LSTM(units=50))
        model.add(Dense(units=1,activation='linear'))                    
        model.summary()

        #Training model with adam optimizer and mean squared error loss function
        model.compile(loss='mean_squared_error',optimizer='adam')
        model.fit(x_train,y_train,validation_data=(x_test,y_test) ,batch_size=64)

        model.save('keras_model.h5')

        #Predicitng on train and test data
        x_predicted = model.predict(x_train)
        y_predicted = model.predict(x_test)

        #Inverse transform to get actual value
        x_predicted =scaler.inverse_transform(x_predicted)
        y_predicted = scaler.inverse_transform(y_predicted)

        
        test = np.vstack((x_predicted,y_predicted))


        #this graph as output
       #Combining the predited data to create uniform data visualization
        plt.plot(scaler.inverse_transform(ds_scaled))
        plt.plot(test)
        plt.title('Actual and Predicted price')
        
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        PredImg = base64.b64encode(img.getvalue()).decode('utf8')

        

        #Getting the last 100 days records
        fut_inp = ds_test[270:]

        fut_inp = fut_inp.reshape(1,-1)

        tmp_inp = list(fut_inp)

        fut_inp.shape

        # #Creating list of the last 100 data
        tmp_inp = tmp_inp[0].tolist()

        #Creating a dummy plane to plot graph one after another
        plot_new=np.arange(1,101)
        plot_pred=np.arange(101,131)

        ds_new = ds_scaled.tolist()

        len(ds_new)

        #Entends helps us to fill the missing value with approx value

        #this graph as output
        #Creating final data for plotting
        final_graph = scaler.inverse_transform(ds_new).tolist()

        #Plotting final results with predicted value after 30 Days
        plt.plot(final_graph,)
        plt.ylabel("Price")
        plt.xlabel("Time")
        plt.title(" Next 30 days Prediction of Opening price of {0} ".format(user_input))
        plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
        plt.legend()

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        HundImg = base64.b64encode(img.getvalue()).decode('utf8')

        # graph kese bhejte hai
        print(openImg, closeImg, PredImg, HundImg)
        return render_template('stock.html', openImg=openImg, closeImg=closeImg, PredImg=PredImg, HundImg=HundImg)
        
    return render_template('stock.html')


if __name__=="__main__":
    app.run(debug=True)