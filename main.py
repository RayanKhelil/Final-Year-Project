import app
import time
import tkinter as tk
import logging


#TODO custom tkinter implementation
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
#TODO format better (show options/formatting, use drop downs and multiple choice...)

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
        #TODO model creation. 

        labelTicker = tk.Label(self, text = "Input ticker: ")
        trainTicker = tk.Text(self, height=1, width=12)
        def get_train_ticker():
            tempTicker = trainTicker.get("1.0", 'end-1c')
            tempTicker = tempTicker.upper()
            return tempTicker
        
        labelTimeWindow = tk.Label(self, text = "Input time window: ")
        timeWindow = tk.Text(self, height=1, width=12)
        def get_time_window():
           tempTimeWindow = int(timeWindow.get("1.0", 'end-1c'))
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
        
        labelToken = tk.Label(self, text = "Use tokenisation?")
        trainToken = tk.Text(self, height=1, width=12)
        def get_token_algo():
            tempToken = trainToken.get("1.0", 'end-1c')
            return tempToken
        

        # def get_scale_type():

        # def get_sma():

        # def get_price():

        # def get_model_algo():

        # def get_class_input():

        # def get_model_name():


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
        
        
        labelTicker.grid(row=2, column = 0)
        useTicker.grid(row=3, column=0)
        labelTimeWindow.grid(row=2, column = 1)
        useTimeWindow.grid(row=3, column=1)
        labelStartDate.grid(row=2, column = 2)
        useStartDate.grid(row=3, column=2)
        labelEndDate.grid(row=2, column = 3)
        useEndDate.grid(row=3, column=3)
        labelResultName.grid(row=4, column = 0)
        resultName.grid(row=5, column=0)
        labelResultNameList.grid(row=4, column = 1)
        resultNameList.grid(row=5, column=1)

        btnPrice.grid(row=6, column=0)
        btnPlot.grid(row=6, column=1)
        labelResultDict.grid(row=6, column=2)

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