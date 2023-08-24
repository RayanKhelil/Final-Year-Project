import app
import time
import tkinter as tk
import logging


#TODO custom tkinter implementation
#TODO do correct widget types
class AppGUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.geometry("1280x720")
        self.title("SPMCT")
        #self.label = tk.Label(self, text="Hello bingle!", font=('Arial', 18))
        #self.label.pack(padx=20, pady=20)
        self._frame = None
        self.switch_frame(HomePage)

    def switch_frame(self, frame_class):
        """Destroys current frame and replaces it with a new one."""
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

  
#TODO validate/prepare inputs in get_functions. catch specicific exceptions to do relevant response (Showing error to user, can overlay text on grid)

class HomePage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        display_hotbar(self, master, bg_hm='#00FF00')
        #TODO home descriptions/introductory stuf


        self.pack(fill='x')
class SentimentPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        display_hotbar(self, master, bg_sa='#00FF00')
        #TODO sentiment funcitons, hella similar to usage (call same app functions for price and plot, new one for sentiment)
        #TODO custom lexicon uploads

        self.pack(fill='x')

class CreationPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        display_hotbar(self, master, bg_mc='#00FF00')
        #TODO Model gets

        labelTicker = tk.Label(self, text = "Input ticker: ")
        trainTicker = tk.Text(self, height=1, width=12)
        def get_train_ticker():
            tempTicker = trainTicker.get("1.0", 'end-1c')
            tempTicker = tempTicker.upper()
            return tempTicker
        
        labelTimeWindow = tk.Label(self, text = "Input time window: ")
        trainTimeWindow = tk.Text(self, height=1, width=12)
        def get_time_window():
           tempTimeWindow = int(trainTimeWindow.get("1.0", 'end-1c'))
           return tempTimeWindow
        
        labelStartDate = tk.Label(self, text = "Input start date: ")        
        trainStartDate = tk.Text(self, height=1, width=12)       
        def get_train_start_date():
            tempStartDate = trainStartDate.get("1.0", 'end-1c')
            return tempStartDate
        
        labelEndDate = tk.Label(self, text = "Input end date: ") 
        trainEndDate = tk.Text(self, height=1, width=12)
        def get_train_end_date():
            tempEndDate = trainEndDate.get("1.0", 'end-1c')
            return tempEndDate

        labelBlacklist = tk.Label(self, text = "Enter blacklist: ")
        trainBlacklist = tk.Text(self, height=1, width=16)
        def get_train_media_blacklist():
            tempBlacklist = trainBlacklist.get("1.0", 'end-1c')
            return tempBlacklist
        
        labelSentimentCalc = tk.Label(self, text = "Enter sentiment calculation method:")
        trainSentimentCalc= tk.Text(self, height=1, width=12)
        def get_sentiment_calc():
            tempSentimentCalc = trainSentimentCalc.get("1.0", 'end-1c')
            tempSentimentCalc = tempSentimentCalc.split(",")
            return tempSentimentCalc
        
        labelToken = tk.Label(self, text = "Use tokenisation?")#RADIO
        trainToken = tk.Text(self, height=1, width=12)
        def get_token_algo():
            tempToken = trainToken.get("1.0", 'end-1c')
            return tempToken
        
        labelScale = tk.Label(self, text = "Enter scaletype")#RADIO
        trainScale = tk.Text(self, height=1, width=12)
        def get_scale_type():
            tempScale = trainScale.get("1.0", 'end-1c')
            return tempScale

        labelSma = tk.Label(self, text= "Enter simple moving averages")
        trainSma = tk.Text(self, height=1, width=12)
        def get_sma():
            tempSma = trainSma.get("1.0", 'end-1c')
            return tempSma
        
        labelPrice = tk.Label(self, text= "Enter pricerange window")
        trainPrice = tk.Text(self, height=1, width=12)
        def get_price():
            tempTPrice = trainPrice.get("1.0", 'end-1c')
            return tempTPrice

        labelModelAlgo = tk.Label(self, text= "Enter ModelAlgo")
        trainModelAlgo = tk.Text(self, height=1, width=12)
        def get_model_algo():
            tempModelAlgo = trainModelAlgo.get("1.0", 'end-1c')
            return tempModelAlgo

        labelClasses = tk.Label(self, text="Enter price classes")
        trainClasses = tk.Text(self, height=1, width=12)
        def get_class_input():
            tempClasses = trainClasses.get("1.0", 'end-1c')
            return tempClasses

        labelModelName = tk.Label(self, text= "Enter model name")
        trainModelName = tk.Text(self, height=1, width=12)
        def get_model_name():
            tempModelname = trainModelName.get("1.0", 'end-1c')
            return tempModelname


        #TODO Buttons to use above inputs


        #grid them start row=2
        labelTicker         .grid(row=2,column=0)
        trainTicker         .grid(row=3,column=0)
        labelTimeWindow     .grid(row=2,column=1)
        trainTimeWindow     .grid(row=3,column=1)
        labelStartDate      .grid(row=2,column=2)
        trainStartDate      .grid(row=3,column=2) 
        labelEndDate        .grid(row=2,column=3)
        trainEndDate        .grid(row=3,column=3)
        labelBlacklist      .grid(row=4,column=0)
        trainBlacklist      .grid(row=5,column=0)
        labelSentimentCalc  .grid(row=4,column=1)
        trainSentimentCalc  .grid(row=5,column=1)
        labelToken          .grid(row=4,column=2)
        trainToken          .grid(row=5,column=2)

        labelScale          .grid(row=6,column=0)
        trainScale          .grid(row=7,column=0)
        labelSma            .grid(row=6,column=1)
        trainSma            .grid(row=7,column=1)
        labelPrice          .grid(row=6,column=2)
        trainPrice          .grid(row=7,column=2)
        labelModelAlgo      .grid(row=6,column=3)
        trainModelAlgo      .grid(row=7,column=3)
        labelClasses        .grid(row=8,column=0)
        trainClasses        .grid(row=9,column=0)
        labelModelName      .grid(row=8,column=1)
        trainModelName      .grid(row=9,column=1)

        self.pack(fill='x')
        
class UsagePage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.modelDict = {}
        self.resultDict = {}
        display_hotbar(self, master, bg_mu='#00FF00')
        
        
        #TODO model usage to get model results
        labelTicker = tk.Label(self, text = "Input ticker: ")
        useTicker = tk.Text(self, height=1, width=12)
        def get_use_ticker():
            tempTicker = useTicker.get("1.0", 'end-1c')
            tempTicker = tempTicker.upper()
            return tempTicker
        
        labelTimeWindow = tk.Label(self, text = "Input time window: ")
        useTimeWindow = tk.Text(self, height=1, width=12)
        def get_time_window_usage():
           tempTimeWindow = int(useTimeWindow.get("1.0", 'end-1c'))
           return tempTimeWindow
        
        labelStartDate = tk.Label(self, text = "Input start date: ")        
        useStartDate = tk.Text(self, height=1, width=12)       
        def get_use_start_date():
            tempStartDate = useStartDate.get("1.0", 'end-1c')
            return tempStartDate
        
        labelEndDate = tk.Label(self, text = "Input end date: ") 
        useEndDate = tk.Text(self, height=1, width=12)
        def get_use_end_date():
            tempEndDate = useEndDate.get("1.0", 'end-1c')
            return tempEndDate
        
        labelResultName = tk.Label(self, text = "Input result name: ")
        resultName = tk.Text(self, height=1, width=12)
        def get_result_name():
            tempName = resultName.get("1.0", 'end-1c')
            logging.info(type(tempName))
            return tempName

        labelResultNameList = tk.Label(self, text = "Input result list: ")
        resultNameList = tk.Text(self, height=1, width=12)
        def get_result_name_list():
            tempNameList = resultNameList.get("1.0", 'end-1c')
            tempNameList = tempNameList.replace(" ","")
            tempNameList = tempNameList.split(",")
            #print(tempNameList)
            return tempNameList

        
        #also logging has issue converting smth to string, what and why?

        btnPrice = tk.Button(self, text="Get price results", command=lambda: app.get_price_results(resultName=get_result_name(), priceTicker=get_use_ticker(), priceTimeWindow=get_time_window_usage(), priceStartDate=get_use_start_date(), priceEndDate=get_use_end_date()))
        btnPlot = tk.Button(self, text="Plot results", command=lambda: app.plot_results(resultNameList = get_result_name_list()))
        
        labelResultDict = tk.Label(self, text = app.resultDict.keys())
        
        
        labelTicker         .grid(row=2, column=0)
        useTicker           .grid(row=3, column=0)
        labelTimeWindow     .grid(row=2, column=1)
        useTimeWindow       .grid(row=3, column=1)
        labelStartDate      .grid(row=2, column=2)
        useStartDate        .grid(row=3, column=2)
        labelEndDate        .grid(row=2, column=3)
        useEndDate          .grid(row=3, column=3)
        labelResultName     .grid(row=4, column=0)
        resultName          .grid(row=5, column=0)
        labelResultNameList .grid(row=4, column=1)
        resultNameList      .grid(row=5, column=1)

        btnPrice            .grid(row=6, column=0)
        btnPlot             .grid(row=6, column=1)
        labelResultDict     .grid(row=6, column=2)

        self.pack(fill='x')

def display_hotbar(self, master, bg_hm = '#EEEEEE', bg_sa = '#EEEEEE', bg_mc = '#EEEEEE', bg_mu = '#EEEEEE'):
    self.columnconfigure(0, weight=1)
    self.columnconfigure(1, weight=1)
    self.columnconfigure(2, weight=1)
    self.columnconfigure(3, weight=1)
    tk.Button(self, text="Home", bg=bg_hm, command=lambda: master.switch_frame(HomePage)).grid(row=1, column = 0, sticky=tk.W+tk.E)
    tk.Button(self, text="Sentiment Analysis", bg=bg_sa, command=lambda: master.switch_frame(SentimentPage)).grid(row=1, column = 1, sticky=tk.W+tk.E)
    tk.Button(self, text="Model Creation", bg=bg_mc, command=lambda: master.switch_frame(CreationPage)).grid(row=1, column = 2, sticky=tk.W+tk.E)
    tk.Button(self, text="Model Usage", bg=bg_mu, command=lambda: master.switch_frame(UsagePage)).grid(row=1, column = 3, sticky=tk.W+tk.E)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gui = AppGUI()
    gui.mainloop()