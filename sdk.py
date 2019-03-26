import tkinter as tk
from functools import partial
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import quandl
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler

import tweepy
from textblob import TextBlob

from bs4 import BeautifulSoup as soup
import requests
import re
from textblob import TextBlob 

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
%matplotlib inline


def sp2(x):
    return(sum(x)/(len(x)+0.1))

def tsa(company):
    import tweepy
    from textblob import TextBlob

    consumer_key='E94sIzRD7vzERsetUTkEO1BdX'
    consumer_secret='xOKQywPpJeZGzI0dB3LB8nmOlMcR7tapUsWTfAh4MsFkaaKOC3'

    access_token='1433534281-hqKtWpz29AyxjmINhv7P5TKdqhnjBCLQAD06IDF'
    access_secret='j9YdOcy52UjJYpWfYEbSL2Mkf789Ix1J3WB7ovqew0Ye7'

    auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_token,access_secret)

    api=tweepy.API(auth)

    public_tweets = api.search(company)
    x=[]
    for tweets in public_tweets:
        #print(tweets.text)
        analysis=TextBlob(tweets.text)
        x.append([analysis.sentiment[0],analysis.sentiment[1]])
    return x

def livemint(url_list):
    text=[]
    for i in url_list:
        if('livemint'  in i):
            page1=requests.get(i)
            htmlpage1 = soup(page1.text,'html.parser')
            no1 = htmlpage1.findAll('script',{})
            l=str(no1[2]) 
            text.append(l[740:-740])
        else:
            continue
    #print(text)
    return text

def et(url_list):
    text=[]
    for i in url_list:
        if('www.economictimes' in i):
            #print(i)
            page1=requests.get(i)
            htmlpage1 = soup(page1.text,'html.parser')
            no1 = htmlpage1.findAll('div',{})
            x=re.search('"Normal"', str(no1)).start()
            y=re.search('<div class="clr"></div><div class="clr">', str(no1)).start()
            l=str(no1)
            l=l[x:y]
            l= soup(l).text
            text.append(l)
    return text

def moneycontrol(url_list):
    text=[]
    for i in url_list:
        if('moneycontrol' in i):
            page1=requests.get(i)
            htmlpage1 = soup(page1.text,'html.parser')
            no1 = htmlpage1.findAll('p',{})
            no1=no1[:-4]
            l=str(no1)
            l=soup(l).text
            text.append(l)
    return text

def nsa(company):
    url_list=[]
    pgno=1
    while(True):
        url = "https://www.ibbi.gov.in/media/media-coverage?title="+company+"&date=&page="+str(pgno)
        page = requests.get(url)
        html_soup = soup(page.text,'html.parser')
        no = html_soup.findAll('td',{})
        if(no ==[]):
            break
        k=1
        for i in range(len(no)//2):
            x=re.search("(?P<url>https?://[^\s]+)", str(no[k])).group("url")
            x=x[:-5]
            url_list.append(x)
            k=k+2
        pgno+=1

    news=livemint(url_list)
    news += et(url_list)
    news += moneycontrol(url_list)
    sa=[]
    for n in news:
        senti=TextBlob(n)
        sa.append([senti.sentiment[0],senti.sentiment[1]])
    return sa

    
def get_from_nsa(company):
    SA=nsa(company)
    if(len(SA)==0):
        return 0,0
    sd1 = np.std([i[0] for i in SA])
    sd2 = np.std([i[1] for i in SA])
    return sd1,sd2

def get_from_tsa(company):
    x=tsa(company)
    if(len(x)==0):
        return 0,0
    sd1=np.std([i[0] for i in x ])
    sd2=np.std([i[1] for i in x ])
    return sd1,sd2


def spl(xx):
    sc1=(int(xx[-1])-int(xx[0]))/int(xx[0])
    return sc1  


d={}
with open('./BSE_metadata.csv', mode='r') as infile:
    reader = csv.reader(infile)
    d={rows[1]:rows[0] for rows in reader}


def call_find(rl1,name,amt):
    name=name.get()
    amt=amt.get()
    sc1=0
    avg=0
    
    if(name in d):
        code=d[name]
    else:
        rl1.config(text="Company Data not found")
        return
    #main model
    dataset = quandl.get("BSE/"+str(code) , authtoken="ySWkR96uDb-6XqvsY-ds")
    index = int(len(dataset)*0.8)
    #CSV1
    dataset_train = dataset[:index]
    #CSV2
    dataset_test = dataset[index+1:]

    # 0:1 return array of shape (x,1)  
    # simply using 0 will return a shape (x,)
    training_set = dataset_train.iloc[:,0:1].values
    real_stock_price = dataset_test.iloc[:,0:1].values

    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    # Creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []
    for i in range(60, len(dataset_train)):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # Initialising the RNN
    regressor = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))

    # Adding the output layer
    regressor.add(Dense(units = 1))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)

    inputs = list(real_stock_price)#(real_stock_price[len(real_stock_price)-60:])
    op=[]
    for i in range(180):
        inputs = (sc.transform(inputs[:]))
        X_test=[]
        for j in range(60+i, 240+i):
            X_test.append(inputs[j-60:j, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = regressor.predict(X_test)
        inputs=np.append(inputs,np.array(predicted_stock_price))
        inputs=np.reshape(inputs,(inputs.shape[0],1))
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        op.append(predicted_stock_price)

    xx=op[0]
    
    sc1=spl(xx)
    
    RETURN = [sc1]
    sd1a,sd2a=get_from_nsa(company)
    sd1b,sd2b=get_from_tsa(company)
    RETURN.extend([sd1a,sd2a,sd1b,sd2b])
    
    return RETURN
    




    
    #PARENT MODEL HERE
    '''
    while(sc1>1):
        sc1-=(2*(sc1)/100)
    sc2=ana(name)
    avg=int(sum(xx)/len(xx))
    if(int(amt)*180<avg):
        sc1/=2
    paap=0
    if((((sc1*64)+(sc2*36))/100)<0.23):
        paap=1
    '''
    if(paap):
        rl1.config(text="Default")
    else :
        rl1.config(text="No default according to our prediction prob = %f" % sc1)
    return










root = tk.Tk()
root.geometry('500x150+150+250')
root.title('Loan Default Prediction')
root.configure(background='#09A3BA')
root.resizable(width=False, height=False)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)

var = tk.StringVar()
amt=tk.StringVar()
input_label1 = tk.Label(root, text="Enter Name of Company (we assume it is listed on BSE)", background='#09A3BA', foreground="#FFFFFF")
result_label1 = tk.Label(root, background='#09A3BA', foreground="#FFFFFF")
result_label1.grid(row=5, columnspan=4)
input_entry1 = tk.Entry(root, textvariable=var)
input_label2 = tk.Label(root, text="Enter Amount of the Loan", background='#09A3BA', foreground="#FFFFFF")
input_entry2 = tk.Entry(root, textvariable=amt)
input_label1.grid(row=0)
input_entry1.grid(row=1, column=1)
input_entry2.grid(row=3,column=1)
input_label2.grid(row=2)

result_button = tk.Button(root, text="Convert", command=lambda : call_find(result_label1,  var,amt), background='#09A3BA', foreground="#FFFFFF")
result_button.grid(row=4, columnspan=4)

root.mainloop()
