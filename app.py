import time
import datetime
import string
import logging
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from GoogleNews import GoogleNews as gn

from newspaper import Article
from newspaper import Config
from newspaper import ArticleException
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import cross_validate, KFold
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.metrics import precision_score

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm

#TODO remove prints, or replace with logging (debug)
def bag_of_words(text, custom = False):
    
    #Split by "!ArticleSeperator!" to calc each article in group then do their avg (so not weighted to longer articles)
    #remove punctuation
    text = text.upper()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    #print(text)
    removeList = []
    positiveTally = 0
    negativeTally = 0
    totalTally = 0

    if custom == True:
        pos_df = pd.read_csv(r'/lexicons/CustomPositive.csv')
        neg_df = pd.read_csv(r'/lexicons/CustomNegative.csv')
    else:
        pos_df = pd.read_csv(r'/lexicons/LoughranMcDonaldPositive.csv')
        neg_df = pd.read_csv(r'/lexicons/LoughranMcDonaldNegative.csv')

    for i in text:
        try:
            positiveTally += pos_df.loc[pos_df['Word'] == i]['Positive'].iloc[-1]
            removeList.append(i) #preparing positive words to be removed from list, no need to check if they are negative
            totalTally +=1
        except IndexError:
            pass

    for i in range(0,len(removeList)):    
        text = [word for word in text if word != removeList[i]]    

    for i in text:
        try:
            negativeTally += neg_df.loc[neg_df['Word'] == i]['Negative'].iloc[-1]
            totalTally +=1
        except IndexError:
            pass

    if totalTally != 0: #to avoid dividing by 0
        #range: -1 to 1 
        sentiment = (positiveTally - negativeTally) / totalTally
        #range: 0 to 1
        sentimentScaled = (sentiment/2) + 0.5
    else:
        sentiment = 0
        sentimentScaled = 0
    
    return (sentimentScaled)


def calculate_sentiment(articleText, sentimentCalc):
    #timer
    match sentimentCalc:

        case 'vader':
            analyzer = SentimentIntensityAnalyzer()
            articleText = articleText.replace(",","")
            senti = analyzer.polarity_scores(articleText)
            sentiment = senti['compound']
            sentiment = (sentiment / 2) + 0.5 #scaling to 0 - 1
            return(sentiment)
        
        case 'LoughranMcDonald':
            sentiment = bag_of_words(articleText, custom = False)
            return sentiment
        case 'custom':
            senti = bag_of_words(articleText, custom = True)
            return sentiment
        
class ArticleInBlacklist(Exception):
    '''Raised when article is from a blacklisted media'''
    pass


def create_dataframe(ticker = 'TSLA', startDate = '2020-01-01', endDate = '2020-02-01', timeWindow = 7, sentimentCalc = None, train = True, useVocabulary = {}, scaleType = None,
                     tokenAlgo = None, smaRangeWindows = None, priceRangeWindow = [], useStockMin = None, useStockMax = None, mediaBlacklist = ['Bloomberg.com']): 
    #when testing dont repeatedly get articles, save some and reuse.
    
    stock = yf.Ticker(ticker)
    startDateDT = datetime.datetime.strptime(startDate, '%Y-%m-%d')
    endDateDT = datetime.datetime.strptime(endDate, '%Y-%m-%d')
    timeWindowDT = datetime.datetime.strptime(str(timeWindow), '%d')

    #populate final_df with dates matching timeWindow
    final_df = pd.DataFrame()
    first_df = pd.DataFrame()#sentiment_df
    second_df = pd.DataFrame()#news_df

    user_agent = 'Mozilla/80.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    config = Config()
    config.browser_user_agent = user_agent

    #populate final_df with dates matching timeWindow
    dateList = []
    indexList = []
    tdelta = startDateDT - endDateDT
    #print(tdelta.days)

    tempDatetime = startDateDT + datetime.timedelta(days=timeWindow)
    dateList.append(tempDatetime)
    tempIndexVal = 0
    indexList.append(tempIndexVal)
    
    while tdelta.days < -timeWindow:
        tempDatetime += datetime.timedelta(days=timeWindow)
        dateList.append(tempDatetime) #done first to add initial tempDatetime above
        tempIndexVal += 1
        indexList.append(tempIndexVal)
        tdelta = dateList[-1] - endDateDT
        #print(tdelta.days)
    #print("final_df initialied")
    tempFinalDict = {tempIndexVal}
    #final_df = pd.DataFrame(dateList, index = indexList) #final_df = pd.DataFrame(dateList, columns=['Datetime'])
    final_df['Datetime'] = pd.DataFrame(dateList, index = indexList)
    #print(final_df)
    
    articleMetric = ""
    if tokenAlgo !=None or sentimentCalc != None:
        #formatting dates for googlenews
        #print("Arrived at googlenews")
        startDateGN = datetime.datetime.strptime(startDate, '%Y-%m-%d').strftime('%m-%d-%Y')
        endDateGN = datetime.datetime.strptime(endDate, '%Y-%m-%d').strftime('%m-%d-%Y')

        googlenews = gn(start=startDateGN, end=endDateGN) #'mm-dd-yyyy' format
        googlenews.clear()
        googlenews.search(ticker) 
        newsResult = googlenews.result()
        #print("newsResult: ",newsResult)
        first_df = pd.DataFrame(newsResult) #searches first 10 articles
        loopCount = int(len(final_df)/2)+2 #approx 3.5 articles per row
        #print(loopCount)
        for i in range(2,loopCount): #approx 9 articles searched per iteration, approx 7 of which are used
            googlenews.getpage(i)
            result=googlenews.result()
            first_df=pd.DataFrame(result)

        #print("first_df created")
        #print(first_df)
        mediaRemovedList = []
        exceptTally = 0 #total removed

        second_df_list=[]
        
        for ind in first_df.index:
            dict={}
            article = Article(first_df['link'][ind])#config=config as second argument changes what the exceptions are. sometimes lower sometimes higher.
            try:       ###ERROR HERE, DONT CATCH EVERYTHING, NEED TIMER? AND FOR TESTING JUST USE ONE DOWNLOAD
                
                article.download()
                article.parse()
                
                if first_df['media'][ind] in mediaBlacklist:
                    raise ArticleInBlacklist(first_df['media'][ind] + " is in blacklist")
                
                if article.publish_date is None:
                    raise TypeError("Article missing datetime, from " + first_df['media'][ind] + " on URL " + first_df['link'][ind])
                
                #Converting offset aware datetimes to offset naive
                awareToNaive = datetime.datetime.strftime(article.publish_date, '%Y-%m-%d')
                awareToNaive = datetime.datetime.strptime(awareToNaive, '%Y-%m-%d')
                
                dict['Datetime']=awareToNaive #(article.publish_date).strptime(startDate, '%Y-%m-%d') #is this okay?
                dict['Title']=article.title
                dict['Article']=article.text
                
                second_df_list.append(dict)

            except ArticleInBlacklist:
                exceptTally+=1
            except (TypeError, ArticleException):
                mediaRemovedList.append(first_df['media'][ind])#to see what medias are being removed other than blacklisted ones
                exceptTally+=1
            

        articleMetric = f"{len(first_df)-exceptTally} / {len(first_df)} articles processed"

        second_df=pd.DataFrame(second_df_list)
        #print(mediaRemovedList)
        
        first_df.to_csv('first_df.csv')
        second_df.to_csv('second_df_unsorted.csv')

        second_df.sort_values(by=['Datetime'], inplace=True)
        second_df = second_df.reset_index(drop=True)
        
        second_df.to_csv('second_df.csv')
        
        #creating new column in final_df, where all article text (title + body) within timewindow is collated.
        final_df['GroupedText']=""
        
        for j in second_df.index:
            for i in final_df.index:
                #print("hi")
                if ((second_df['Datetime'][j] - final_df['Datetime'][i]).days) < 0: #LOOP SECOND_DF FIRST, THEN LOOP FINAL_DF WITHIN
                    #add "!ArticleSeperator!" when adding new article. so can split in bag of words, and calc avg sentiment of articles
                    final_df.loc[i, 'GroupedText'] += " " + second_df.loc[j, 'Title'] + " " + second_df.loc[j, 'Article'] ##ERROR '[8] not in index', 
                    break
        
        #print(final_df.head())

        ###Sentiment
        if sentimentCalc != None:
            final_df['Sentiment']=""
            for row in final_df.index:
                final_df.loc[row, 'Sentiment'] = calculate_sentiment(final_df.loc[row, 'GroupedText'], sentimentCalc)
                #final_df['Sentiment'][row]
        

        ###Tokenization
        tokenCount = 10 #number of words to be put as attributes
        
        if tokenAlgo != None:
            if train == True:
                vocab = None
            else:
                vocab = useVocabulary
                

            vr = TfidfVectorizer(max_df=0.4, min_df=0.1, max_features = tokenCount, stop_words='english', vocabulary = vocab)#if using, set vocab to input
            vr_df = vr.fit_transform(final_df['GroupedText'])# Create Dataframe from TF-IDFarray
            tfidf_df = pd.DataFrame(vr_df.toarray(), columns=vr.get_feature_names_out())

            #return this value if training, so model can be consistent in what words become attributes
            useVocabulary = vr.vocabulary_

            for word in useVocabulary:
                final_df[word] = tfidf_df[word]
        else:
            useVocabulary = None

    
    ###Adding prices

    if train == True:
        if scaleType == 'minmax' and smaRangeWindows != None:
            ##Calculating values to scale each price by
            #Prices may need to be collected from before and after the start and end dates
            
            if (smaRangeWindows[0][0] - smaRangeWindows[0][1]) < 0:
                stockStartDT = startDateDT + datetime.timedelta(days=(smaRangeWindows[0][0])-smaRangeWindows[0][1])
            else:
                stockStartDT = startDateDT 
        
            stockEndDT = endDateDT + datetime.timedelta(days=smaRangeWindows[-1][0])

            stockPrice_df = stock.history(start=stockStartDT, end=stockEndDT)
            #values for scaling
            stockMin = stockPrice_df["Close"].min()
            stockMax = stockPrice_df["Close"].max()

            useStockMin = stockMin
            useStockMax = stockMax
            
        else:
            useStockMin = None
            useStockMax = None
    else:
        stockMin = useStockMin
        stockMax = useStockMax
    if scaleType == 'minmax':
        stockDif = stockMax - stockMin

    

    tempPrice=0
    

    #print(stockMin, stockMax)
    #print(final_df)

    final_df['DatePrice'] = ""

    for row in final_df.index:
            tempStart = final_df['Datetime'][row] - datetime.timedelta(days=timeWindow)
            tempEnd = final_df['Datetime'][row]
            tempPrice_df = stock.history(start = tempStart, end = tempEnd)
            if not tempPrice_df.empty:
                tempPrice = tempPrice_df["Close"].mean()
            #if scaleType == 'minmax':
            #    tempPrice = (tempPrice-stockMin)/stockDif
            final_df.loc[(row, 'DatePrice')] = tempPrice

    ##ONLY DO IF NOT EMPTY/NONE
    if smaRangeWindows != None:
        for i in range(0,len(smaRangeWindows)):
            tempName = str(smaRangeWindows[i][0]) + " " + str(smaRangeWindows[i][1])
            #print("tempName: ", tempName)
            final_df[tempName] = ""
            for row in final_df.index:
                tempStart = final_df['Datetime'][row] + datetime.timedelta(days=(smaRangeWindows[i][0]) - smaRangeWindows[i][1])
                tempEnd = final_df['Datetime'][row] + datetime.timedelta(days=(smaRangeWindows[i][0]))
                tempPrice_df = stock.history(start = tempStart, end = tempEnd)
                if not tempPrice_df.empty: #will use previous tempPrice value if data is missing(misses for weekends and holidays)
                    tempPrice = tempPrice_df["Close"].mean()
                    if scaleType == 'minmax':
                        tempPrice = (tempPrice-stockMin)/stockDif
                final_df.loc[(row, tempName)] = tempPrice
            
    #print(final_df)
    
    final_df["Prediction"] = ""

    if train == True:
        for row in final_df.index:
            #print(final_df)
            tempStart = final_df['Datetime'][row] + datetime.timedelta(days=(priceRangeWindow[0]) - priceRangeWindow[1])
            tempEnd = final_df['Datetime'][row] + datetime.timedelta(days=(priceRangeWindow[0]))
            tempPrice_df = stock.history(start = tempStart, end = tempEnd)
            if not tempPrice_df.empty:
                tempPrice = tempPrice_df["Close"].mean()
            #if scaleType == 'minmax':
            #    tempPrice = (tempPrice-stockMin)/stockDif
            final_df.loc[(row,'Prediction')] = tempPrice

    ##GroupedText column is not needed anymore
    try:
        final_df = final_df.drop('GroupedText', axis=1)
    except KeyError:
        pass

    

    if train == True:
        #return metrics
        #return number of articles trained on
        return final_df, useVocabulary, useStockMin, useStockMax, articleMetric
    else:
        return final_df
    


class StandardModel():
    def __init__(self, ticker = 'TSLA', startDate = '2021-01-01', endDate = '2021-04-01', timeWindow = 7, sentimentCalc = 'vader', tokenAlgo = True, smaRangeWindows = None, priceRangeWindow = [7,7], 
                 modelAlgo='RFC', custom_train = None, scaleType = 'minmax', classInput=[0], mediaBlacklist = ['Bloomberg.com']):
        
        self.ticker = ticker
        self.startDate = startDate
        self.endDate = endDate
        self.timeWindow = timeWindow
        self.sentimentCalc = sentimentCalc
        self.tokenAlgo = tokenAlgo
        self.smaRangeWindows = smaRangeWindows
        self.priceRangeWindow = priceRangeWindow
        self.mediaBlacklist = mediaBlacklist

        self.modelAlgo = modelAlgo
        self.custom_train = custom_train
        self.scaleType = scaleType
        self.classInput = classInput

        self.train()
        
    def train(self): #Allows for custom training df. stretch: it would also allow for training with user set df(but they must also do their own usage df) (they can construct these dfs in the app itself)
        '''train model, produce metrics'''
        timeStart = time.perf_counter()
        if self.custom_train == None:
            returnStuff = create_dataframe(train=True, ticker=self.ticker, sentimentCalc=self.sentimentCalc, tokenAlgo=self.tokenAlgo, smaRangeWindows=self.smaRangeWindows, scaleType=self.scaleType,
                                           priceRangeWindow = self.priceRangeWindow, startDate = self.startDate, endDate = self.endDate, timeWindow=self.timeWindow, mediaBlacklist=self.mediaBlacklist)

        else:
            returnStuff = self.custom_train

        if isinstance(returnStuff, tuple): 
            training_df, self.vocab, self.useStockMin, self.useStockMax, self.articleMetric = returnStuff
        else:
            training_df = returnStuff

        Xy_df = training_df.drop(['Datetime','DatePrice'], axis = 1)
        X = Xy_df.drop('Prediction', axis = 1)
        y = Xy_df['Prediction'].copy()

        if self.modelAlgo == 'RFC' or self.modelAlgo == 'SVM': #or any other class algo
            self.priceClasses = self.create_class_names(self.classInput)

            for i in training_df.index:
                y.loc[i] = self.calculate_price_class(training_df.loc[i, 'Prediction'], training_df.loc[i, 'DatePrice'], self.classInput)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        if self.scaleType == 'sc':
            self.sc = StandardScaler()
            X_train = self.sc.fit_transform(X_train)
            X_test = self.transform(X_test)

        #before or after above?
        X_train = X_train.astype('float')
        X_test = X_test.astype('float')
        y_train = y_train.astype('float')
        y_test = y_test.astype('float')
        
        match self.modelAlgo:
 
            case 'SVR':#regression
                self.model = SVR(kernel='rbf')
                self.model.fit(X_train, y_train)
                trainPred = self.model.predict(X_test)
                timeEnd = time.perf_counter() - timeStart
                self.time = (f"Training took: {timeEnd}s")

                self.metric = r2_score(y_test, trainPred)

            case 'RFC':#classification
                self.model = RandomForestClassifier(n_estimators=200)
                self.model.fit(X_train, y_train)
                trainPred = self.model.predict(X_test)
                timeEnd = time.perf_counter() - timeStart
                self.time = (f"Training took: {timeEnd}s")

                self.metric = classification_report(y_test, trainPred)

            case 'SVM':#classification
                self.model = svm.SVC()
                self.model.fit(X_train, y_train)
                trainPred = self.model.predict(X_test)
                timeEnd = time.perf_counter() - timeStart
                self.time = (f"Training took: {timeEnd}s")

                self.metric = classification_report(y_test, trainPred)
     
    def use(self, useStartDate, useEndDate, custom_use_df = None):
        '''Uses model on a dataframe set by the user, and returns dataframe of predicted prices with matching datetimes (not offset)'''
        #totalUseTime
        if custom_use_df == None:
            use_df = create_dataframe(train=False, useVocabulary=self.vocab, ticker=self.ticker, sentimentCalc=self.sentimentCalc, tokenAlgo=self.tokenAlgo, smaRangeWindows=self.smaRangeWindows, priceRangeWindow=self.priceRangeWindow, 
                                    startDate=useStartDate, endDate=useEndDate, scaleType=self.scaleType, mediaBlacklist=self.mediaBlacklist, useStockMin=self.useStockMin, useStockMax=self.useStockMax, timeWindow=self.timeWindow)
        else:
            use_df = custom_use_df
        #if custom_use_df != None then you know its specialusage: return the single value, still scaled

        #drop unneeded columns (prediction column is empty here)
        X_use = use_df.drop(['Datetime','DatePrice','Prediction'], axis = 1)
        
        #if self.scaleType == 'sc' then do sc using self.sc set in train
        if self.scaleType == 'sc':
            X_use = self.sc.transform(X_use)

        #prediction will be a class instead of a continuous price for classification models
        usePred = self.model.predict(X_use)
        
        #format prediction using datetime from use_df, unscale, and if clasification use classification attributes for low high class
            #create dataframe of right dates, then just match prediction to it, SEE HOW I INITIALISED final_df in create_dataframe()

        result_df = pd.DataFrame()
        dateList = []
        indexList = []
        tempIndexVal = 0 

        for index in use_df.index:
            predDate = use_df.loc[index, 'Datetime'] + datetime.timedelta(days = self.priceRangeWindow[0])
            dateList.append(predDate)
            indexList.append(tempIndexVal)
            tempIndexVal += 1

        result_df['Datetime'] = pd.DataFrame(dateList, index = indexList)
        result_df['Prediction'] = usePred

        if self.modelAlgo == 'RFC' or self.modelAlgo == 'SVM':#if classification
            #Add low and high
            result_df['Low'] = ""
            result_df['High'] = ""
            for index in result_df.index:
                result_df.loc[index, 'Low'] = self.calculate_real_low(self.classInput, use_df.loc[index, 'DatePrice'], result_df.loc[index, 'Prediction'])
                result_df.loc[index, 'High'] = self.calculate_real_high(self.classInput, use_df.loc[index, 'DatePrice'], result_df.loc[index, 'Prediction'])


            result_df['DatetimeStart'] = [dt-datetime.timedelta(days=self.timeWindow) for dt in result_df['Datetime']]

            
        #return result_df
        return(result_df)
    
    def create_class_names(self, classes):
        '''Takes in price classes list, returns list with formatted names'''
      
        nameClasses = []
        nameClasses.append(f'x<{classes[0]}%')#first class is unique
        if len(classes) == 1:
            nameClasses.append(f'{classes[0]}%<x')
            return(nameClasses)
            
        for j in range(1,len(classes)):
            nameClasses.append(f'{classes[j-1]}%<=x<{classes[j]}%')
            
        nameClasses.append(f'{classes[-1]}%<=x')#last class is unique
        return(nameClasses)



    def calculate_price_class(self, newP, oldP, classes):
        '''converts price to index of its matching class in nameClasses'''
        
        priceChange=100*((newP-oldP)/newP)
        if priceChange > classes[-1]:
            return len(classes)
        for i in range(len(classes)):
            if priceChange < classes[i]:
                return i
            
    def calculate_real_low(self, classInp, startPrice, index):
        if index == 0:
            return 0

        for i in range(1, len(classInp)+1):
            if index == i:
                return startPrice * (1 + (classInp[i-1]/100))

    def calculate_real_high(self, classInp, startPrice, index):
        if index == len(classInp):
            return 0

        for i in range(0, len(classInp)):
            if index == i:
                return startPrice * (1 + (classInp[i]/100))
            

#TODO remove these functions, then remove default args of functions below
###Model Creation settings
def get_train_ticker():
    trainTicker='AMZN'#would = html doc/div
    #clean/validate/raise an error on value
    return trainTicker

def get_train_start_date():
    trainStartDate = '2021-11-01'
    #check if valid start(try strptime)
    return trainStartDate

def get_train_end_date():
    trainEndDate = '2022-08-01'
    #check if valid end
    return trainEndDate

def get_train_media_blacklist():
    mediaBlacklist = ['Bloomberg.com']
    return mediaBlacklist

def get_time_window():
    timeWindow = 7
    return timeWindow

def get_sentiment_calc():
    sentimentCalc = None
    return sentimentCalc

def get_token_algo():
    tokenAlgo = None
    return tokenAlgo

def get_scale_type():
    scaleType = 'minmax'
    return scaleType

def get_sma():
    smaRangeWindows=[[0,7],[7,7]]
    return smaRangeWindows

def get_price():
    priceRangeWindow = [14,7]
    return priceRangeWindow

def get_model_algo():
    modelAlgo = 'RFC'
    return modelAlgo

def get_class_input():
    classInput = [-5,0,5]
    return classInput

def get_model_name():
    modelName = "TestSMMulti"
    return modelName

###Model Usage Settings
def get_use_ticker():
    trainTicker='AMZN'#would = html doc/div
    #clean/validate/raise an error on value
    return trainTicker

def get_time_window_usage():
    timeWindow = 7
    return timeWindow

def get_model_name_usage():
    modelName = "TestSMMulti"
    return modelName

def get_use_start_date():
    useStartDate = '2022-09-01'
    #check if valid start(try strptime)
    return useStartDate

def get_use_end_date():
    useEndDate = '2023-05-01'
    #check if valid end
    return useEndDate

def get_result_name():
    result_name = 'Result2'
    return result_name

def get_result_name_list():
    result_name = ['Result1', 'Result2']
    return result_name


modelDict = {}
resultDict = {}
##Buttons
def create_model(name, ticker, startDate, endDate, mediaBlacklist, timeWindow, sentimentCalc, tokenAlgo, scaleType, smaRangeWindows, priceRangeWindow, modelAlgo, classInput):
    
    sm = StandardModel(ticker = ticker, startDate=startDate, endDate=endDate, mediaBlacklist=mediaBlacklist, timeWindow=timeWindow, sentimentCalc=sentimentCalc, tokenAlgo=tokenAlgo,
                        scaleType=scaleType, smaRangeWindows=smaRangeWindows, priceRangeWindow=priceRangeWindow, modelAlgo=modelAlgo, classInput=classInput)
    modelDict.update({name: sm})
    
def show_metrics(name):
    print(modelDict[name].metric)
    print("\n"+modelDict[name].time)
    print("\n"+modelDict[name].articleMetric)
    
def get_model_results(resultName, name, useStartDate, useEndDate):
    temp_df = modelDict[name].use(useStartDate, useEndDate)
    resultDict.update({resultName: temp_df})

def get_price_results(resultName, priceTicker, priceTimeWindow, priceStartDate, priceEndDate):
    ###Force usage of string? datetime doesnt work for 1 day windows unlike string (this is done in create_dataframe)
    print(priceTicker)
    temp_df = create_dataframe(train=False, ticker=priceTicker, timeWindow=priceTimeWindow, startDate=priceStartDate, endDate=priceEndDate)
    #need just Datetime and Dateprice
    new_df = temp_df[['Datetime','DatePrice']].copy()
    logging.info(resultName)
    resultDict.update({resultName: new_df})

def plot_results(resultNameList=get_result_name_list()):
    #SET SIZE/TIMEFRAME OF GRAPH
    
    #logging.info("list of results being plotted:", resultNameList)
    #TODO label the lines in graph
    for name in resultNameList:
        #logging.info("specific result being plotted:", name)
        temp_df = resultDict[name]
        if 'Low' in temp_df.columns: #classification results
            
            high_df = temp_df.drop(temp_df[temp_df.Low == 0].index)
            low_df = temp_df.drop(temp_df[temp_df.High == 0].index)
            
            endHigh = high_df.loc[:, 'Datetime']
            startHigh = high_df.loc[:, 'DatetimeStart']
            priceHigh = high_df.loc[:, 'Low']

            endLow = low_df.loc[:, 'Datetime']
            startLow = low_df.loc[:, 'DatetimeStart']
            priceLow = low_df.loc[:, 'High']

            plt.plot([endHigh, startHigh], [priceHigh, priceHigh], 'g-')
            plt.plot([endLow, startLow], [priceLow, priceLow], 'r-')

        elif 'DatePrice' in temp_df.columns: #real price results
            plt.plot(temp_df['Datetime'], temp_df['DatePrice'], color='k')
            
        elif 'Prediction' in temp_df.columns: #regression results
            plt.plot(temp_df['Datetime'], temp_df['Prediction'], color='b')

    plt.xlabel("Datetime")
    plt.ylabel("Price")
    plt.show()