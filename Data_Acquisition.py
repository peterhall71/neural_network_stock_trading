###https://gist.github.com/robcarver17/f50aeebc2ecd084f818706d9f05c1eb4
###https://interactivebrokers.github.io/tws-api/classIBApi_1_1EClient.html#a5eac5b7908b62c224985cf8577a8350c
###https://github.com/GerHobbelt/tws_c_api/blob/master/twsapi.h

###Account: U3143307
###Username: pethal711
###Password: PwgZZ44tV5f7

###Data_Acquisition_v2.py

#LIBRARIES
import sys, os, csv, glob, shutil, queue, time, datetime, pathlib2
import numpy as np
import pandas as pd
from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.contract import Contract as IBcontract
from threading import Thread 

#PROGRAM SPECIFICS

DEFAULT_HISTORIC_DATA_ID=50
DEFAULT_GET_CONTRACT_ID=43

## marker for when queue is finished
FINISHED = object()
STARTED = object()
TIME_OUT = object()

#PARAMETERS

#data span settings
Data_Timeframe = "1 Y"
Bar_Interval = '5 mins' #'1 min' '2 mins' '5 mins' '3 mins' '5 mins' '15 mins' '30 mins' '1 hour' '1 day' 
Max_Wait_Time = 600

#contract specifics
Security_Type = "STK"
Last_Trade_Date = "20191005"
Symbols = ['SLB', 'M', 'MU', 'PCG', 'WFC', 'HPQ', 'ZNGA', 'SIRI', 'CSCO', 'NOK', 'VZ', 'CMCSA', 'MRO', 'CTL', 'INTC', 'ORCL']
Exchange = "NYSE"

#"BAC", "AMD", "SNAP", "F", "AAPL", 'ROKU', "S", "NVDA", "FB", "MSFT", "GE"

#TWS port information
IP_Address = "127.0.0.1"
TWS_Port = 7496
ID = 7011

class finishableQueue(object):

    def __init__(self, queue_to_finish):
        self._queue = queue_to_finish
        self.status = STARTED

    def get(self, timeout):
        #Returns a list of queue elements once timeout is finished, or a FINISHED flag is received in the queue
        #:param timeout: how long to wait before giving up
        #:return: list of queue elements
        contents_of_queue=[]
        finished=False

        while not finished:
            try:
                current_element = self._queue.get(timeout=timeout)
                if current_element is FINISHED:
                    finished = True
                    self.status = FINISHED
                else:
                    contents_of_queue.append(current_element)
                    ## keep going and try and get more data
            except queue.Empty:
                ## If we hit a time out it's most probable we're not getting a finished element any time soon
                ## give up and return what we have
                finished = True
                self.status = TIME_OUT
        return contents_of_queue

    def timed_out(self):
        return self.status is TIME_OUT

class TestWrapper(EWrapper):
    #The wrapper deals with the action coming back from the IB gateway or TWS instance
    #We override methods in EWrapper that will get called when this action happens, like currentTime
    #Extra methods are added as we need to store the results in this object

    def __init__(self):
        self._my_contract_details = {}
        self._my_historic_data_dict = {}

    ## error handling code
    def init_error(self):
        error_queue=queue.Queue()
        self._my_errors = error_queue

    def get_error(self, timeout=5):
        if self.is_error():
            try:
                return self._my_errors.get(timeout=timeout)
            except queue.Empty:
                return None
        return None

    def is_error(self):
        an_error_if=not self._my_errors.empty()
        return an_error_if

    def error(self, id, errorCode, errorString):
        ## Overriden method
        errormsg = "IB error id %d errorcode %d string %s" % (id, errorCode, errorString)
        self._my_errors.put(errormsg)

    ## get contract details code
    def init_contractdetails(self, reqId):
        contract_details_queue = self._my_contract_details[reqId] = queue.Queue()
        return contract_details_queue

    def contractDetails(self, reqId, contractDetails):
        ## overridden method
        if reqId not in self._my_contract_details.keys():
            self.init_contractdetails(reqId)
        self._my_contract_details[reqId].put(contractDetails)

    def contractDetailsEnd(self, reqId):
        ## overriden method
        if reqId not in self._my_contract_details.keys():
            self.init_contractdetails(reqId)
        self._my_contract_details[reqId].put(FINISHED)

    ## Historic data code
    def init_historicprices(self, tickerid):
        historic_data_queue = self._my_historic_data_dict[tickerid] = queue.Queue()
        return historic_data_queue

    def historicalData(self, tickerid , bar):
        ## Overriden method
        ## Note I'm choosing to ignore barCount, WAP and hasGaps but you could use them if you like
        bardata=(bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume)

        historic_data_dict=self._my_historic_data_dict

        ## Add on to the current data
        if tickerid not in historic_data_dict.keys():
            self.init_historicprices(tickerid)

        historic_data_dict[tickerid].put(bardata)

    def historicalDataEnd(self, tickerid, start:str, end:str):
        ## overriden method
        if tickerid not in self._my_historic_data_dict.keys():
            self.init_historicprices(tickerid)

        self._my_historic_data_dict[tickerid].put(FINISHED)

class TestClient(EClient):
    #The client method
    #We don't override native methods, but instead call them from our own wrappers

    def __init__(self, wrapper):
        ## Set up with a wrapper inside
        EClient.__init__(self, wrapper)

    def resolve_ib_contract(self, ibcontract, reqId=DEFAULT_GET_CONTRACT_ID):
        #From a partially formed contract, returns a fully fledged version
        #:returns fully resolved IB contract

        ## Make a place to store the data we're going to return
        contract_details_queue = finishableQueue(self.init_contractdetails(reqId))
        print("Getting full contract details from the server... ")

        self.reqContractDetails(reqId, ibcontract)

        ## Run until we get a valid contract(s) or get bored waiting
        MAX_WAIT_SECONDS = 180
        new_contract_details = contract_details_queue.get(timeout = MAX_WAIT_SECONDS)

        while self.wrapper.is_error():
            print(self.get_error())

        if contract_details_queue.timed_out():
            print("Exceeded maximum wait for wrapper to confirm finished - seems to be normal behaviour")

        if len(new_contract_details)==0:
            print("Failed to get additional contract details: returning unresolved contract")
            return ibcontract

        if len(new_contract_details)>1:
            print("got multiple contracts using first one")

        new_contract_details=new_contract_details[0]
        resolved_ibcontract=new_contract_details.contract
        return resolved_ibcontract

    def get_IB_historical_data(self, ibcontract, durationStr=Data_Timeframe, barSizeSetting=Bar_Interval,
                               tickerid=DEFAULT_HISTORIC_DATA_ID):
        #Returns historical prices for a contract, up to today
        #ibcontract is a Contract
        #:returns list of prices in 4 tuples: Open high low close volume

        ## Make a place to store the data we're going to return
        historic_data_queue = finishableQueue(self.init_historicprices(tickerid))

        # Request some historical data. Native method in EClient
        self.reqHistoricalData(
            tickerid,  # tickerId,
            ibcontract,  # contract,
            datetime.datetime.today().strftime("%Y%m%d %H:%M:%S %Z"),  # endDateTime,
            durationStr,  # durationStr,
            barSizeSetting,  # barSizeSetting,
            "TRADES",  # whatToShow,
            1,  # useRTH,
            1,  # formatDate
            False,  # KeepUpToDate <<==== added for api 9.73.2
            [] ## chartoptions not used
            )

        ## Wait until we get a completed data, an error, or get bored waiting
        MAX_WAIT_SECONDS = Max_Wait_Time
        print("Getting historical data from the server... could take %d seconds to complete " % MAX_WAIT_SECONDS)

        historic_data = historic_data_queue.get(timeout = MAX_WAIT_SECONDS)

        while self.wrapper.is_error():
            print(self.get_error())

        if historic_data_queue.timed_out():
            print("Exceeded maximum wait for wrapper to confirm finished - seems to be normal behaviour")

        self.cancelHistoricalData(tickerid)
        return historic_data

class TestApp(TestWrapper, TestClient):
    def __init__(self, ipaddress, portid, clientid):
        TestWrapper.__init__(self)
        TestClient.__init__(self, wrapper=self)
        self.connect(ipaddress, portid, clientid)
        thread = Thread(target = self.run)
        thread.start()
        setattr(self, "_thread", thread)
        self.init_error()

#TWS port information
app = TestApp(IP_Address, TWS_Port, ID)

for stock in Symbols:

    #contract specifics
    ibcontract = IBcontract()
    ibcontract.secType=Security_Type
    ibcontract.lastTradeDateOrContractMonth=Last_Trade_Date
    ibcontract.symbol=stock
    ibcontract.exchange=Exchange
    resolved_ibcontract=app.resolve_ib_contract(ibcontract)

    #convert data into numpy array, remove extraneous rows, convert to float type
    historic_data = np.array(app.get_IB_historical_data(resolved_ibcontract))
    historic_data_trimmed = historic_data[:, 1:5]
    historic_data_trimmed = historic_data_trimmed.astype(np.float)
    
    #save data to csv and delay before next request
    filepath = os.path.join(r'C:\Users\peter\Downloads', stock + '.csv')
    np.savetxt(filepath, historic_data_trimmed, delimiter=",")
    time.sleep(120)
    
app.disconnect()