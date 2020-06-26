###https://scikit-learn.org/stable/modules/clustering.html


###Master.py

#LIBRARIES
import sys, os, shutil, statistics 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc

#PARAMETERS

#initialization
Initialize = False
Raw_Data_Folder = 'Raw_Data_Temp'

#clustering
Clustering = True
Analysis = False
Number_of_Clusters = 15

#exectution
Account_Value = 60000
Order_Percentage = 0.01
Trade_Images = True

#FUNCTIONS & CLASSES


class Cluster:
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.Clustering()
    
    def Clustering(self):
        #create K-Means classifier and fit data
        km = KMeans(
                n_clusters = Number_of_Clusters,
                init = 'k-means++',
                n_init = 10,
                max_iter = 300,
                tol = 1e-04,
                precompute_distances = 'auto',
                verbose = 0,
                random_state=None,
                copy_x = True,
                n_jobs = 4,
                algorithm = 'auto'
                )
        
        print('')
        print('Applying Kmeans Clustering')
        km.fit(self.dataset)
        
        #reshape cluster labels and concatenate
        self.labeled = np.concatenate((self.dataset, np.reshape(km.labels_, (-1, 1))), axis = 1)
        print('')
        print('Total Record Count:', len(self.dataset))
        print('')
        
        #create avg cluster images
        cluster_count = 0
        while cluster_count < Number_of_Clusters:
            #create array for specific cluster based on cluster_count, then remove cluster column, print number of records in cluster
            clus = self.labeled[self.labeled[:,44] == cluster_count]
            clus = np.delete(clus, 44, axis=1)
            print('Record Count,' , cluster_count,':', len(clus))
            
            #find  mean of columns and reformat cluster dataset from one averaged 44 column row to set of 11 ohlc records
            clusavg = clus.mean(axis=0)
            clusarray = np.zeros((11,4))
            count = 0
            while (count*4) < 44:
                #select index section, insert into clussarray, and index count (acting as the startpoint as well
                clusarray[count] = clusavg[:][(count*4):(count*4 + 4)]
                count += 1

            #configure plot, save image, and close
            Candlestick_Plot(clusarray, cluster_count, 'Cluster_Images', cluster_count)
            
            cluster_count += 1
        
        #remove prediction portion of the array
        self.labeled_trimmed = np.delete(self.labeled, np.s_[24:44], axis=1)
        
        #save training_data to csv
        np.savetxt("Execution_Files\Training_Data.csv", self.labeled_trimmed, delimiter=",")
        
        #run analysis if indicated, return both arrays
        if Analysis: self.Cluster_Analysis()
        return

    def Cluster_Analysis(self):
        #create labels and list of cluster plot names
        labels = range(Number_of_Clusters)
        cluster_plot_names = [cluster_plot_names.append('%d' %i) for i in labels]
        
        #split dataset into training data and clusters
        X_analysis = self.labeled_trimmed[:,0:24]
        Y_analysis = self.labeled_trimmed[:,24]

        #split data to train and test on 80-20 ratio
        X_train, X_test, y_train, y_test = train_test_split(X_analysis, Y_analysis, test_size = 0.2, random_state=None)

        #create and train KNN classifier, make predictions on unseen test data
        knn = KNN_Classifier()
        knn.fit(X_train, y_train)
        knn_predictions = knn.predict(X_test)
        
        #returns the coefficient of determination R^2 of the prediction, 1 is perfect prediction and 0 means that there is no linear relationship between X and y
        #confusion matrix
        print("")
        print("Accuracy: {}%".format(classifier.score(X_test, y_test) * 100 ))
        print("")
        cm = confusion_matrix(y_test, clf_predictions, labels)
        print(cm)
        print("")

class Initialization:
    
    def __init__(self):
        self.complete_dataset = []
        self.Load_Files()
        
    def Load_Files(self):
        print('')
        print('Loading Raw Data...')
        
        #read each file from Raw_Data, convert np.array to list of lists, iterate through to create new dataset, convert to numpy array
        for file in os.listdir(r'.\%s' %Raw_Data_Folder):
            raw_data = np.genfromtxt(os.path.join(Raw_Data_Folder, file), delimiter=',')
            raw_data.tolist()
            
            #iterate through numpy array to create new dataset
            for index, i in enumerate(raw_data):
                
                #select subsection, flatten list, check if list has 44 elements, normalize and append to primary_list
                sublist = [item for sublist in raw_data[index: index + 11] for item in sublist]
                
                if len(sublist) <44: break
                
                self.complete_dataset.append([x/statistics.mean(sublist) for x in sublist])
            
            print(file)
        
        self.complete_dataset = np.array(self.complete_dataset)
        np.savetxt("Execution_Files\Primary.csv", self.complete_dataset, delimiter=",")
        
        return

class Test_Data:
    def __init__(self, name):
        self.name = name
        self.array = np.genfromtxt(os.path.join('Test_Files', name + '.csv'), delimiter=',')
        self.main_array = np.zeros((6, 4))
        self.shares = 0
        self.buy_flag = 0
        self.purchase_point = 0
        self.purchase_price = 0
        self.knn_predictions = 0
        return

    def Trade_Loop(self):
        global Account_Value
        #update main_array, select next section to insert into main array, delete last row of main_array, add new array in index 0
        self.main_array = np.delete(self.main_array, 5, axis=0)
        self.main_array = np.insert(self.main_array, 0, self.array[live_counter], axis=0)
        if live_counter < len(self.main_array): return
        
        #PREDICTION LOOP
        if self.shares + self.buy_flag == 0:
            #flatten and normalize main_array, reshape to one row np.array (1,28), make predictions, set buy_flag
            test_flat = self.main_array.flatten(order='C')
            test_norm = np.array([x/statistics.mean(test_flat) for x in test_flat])
            self.knn_predictions = knn.predict(test_norm.reshape(1,-1))
            
            if self.knn_predictions in buy_indicators:
                self.buy_flag = 1
                return
         
        #buy order
        if self.shares + self.buy_flag == 1:
            self.purchase_price = self.main_array[0][3]
            self.shares = (Order_Percentage*Account_Value)/self.purchase_price
            Account_Value = Account_Value*(1 - Order_Percentage)
            self.purchase_point = live_counter

        #SELL LOOP    
        if self.shares > 0:
                
            if live_counter - self.purchase_point == 5:
                #sell
                sell_price = self.main_array[0][3]
                Account_Value = Account_Value + self.shares*sell_price
                sell_point = live_counter
                self.buy_flag = 0

                #trading record
                profit = self.shares*(sell_price - self.purchase_price)
                time_diff = sell_point - self.purchase_point
                trade_record.append([self.purchase_point, self.purchase_price, sell_point, sell_price, time_diff, self.shares, profit, self.knn_predictions])
                
                if Trade_Images:
                    #select section to be plotted, and check if array has 11 records, the last set most likely will not
                    image_section = self.array[live_counter - 10 : live_counter + 1 ,  :]
                    if len(image_section) <11: return
                    Candlestick_Plot(image_section, self.knn_predictions, 'Trade_Images', live_counter)
                
                self.shares = 0
        return

def Delete_Create_Folder(folder):
    #delete folder and create new one
    try:
        shutil.rmtree(folder)
        os.mkdir(folder)
    #just create new folder
    except:
        os.mkdir(folder)
    return

def KNN_Classifier():
    knn = KNeighborsClassifier(
                algorithm = 'auto',
                leaf_size = 30,
                metric = 'minkowski',
                metric_params = None,
                n_jobs = None,
                n_neighbors = 6,
                p = 2,
                weights = 'uniform'
                )
    return knn

def Candlestick_Plot(plot_data, plot_title, Folder_name, file_name):
    #add static date column to array
    array_dates = np.concatenate((np.reshape(np.arange(736619,736630), (-1, 1)), plot_data), axis = 1)
    
    #configure plot, save image and close
    f1, ax = plt.subplots(figsize = (10,5))
    candlestick_ohlc(ax, array_dates, width=0.6, colorup='green', colordown='red')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.title(plot_title)
    plt.savefig(os.path.join(Folder_name, '%d.png' %file_name), bbox_inches='tight')
    plt.close()

def Input_and_Convert(message):
    while True:
        try:
            #take comma delimited input and convert to list of integers
            print('')
            cluster_list = input(message)
            cluster_list = cluster_list.split(',')
            cluster_list = [int(x.strip()) for x in cluster_list]
            break
        except:
            print('Invalid entry, please try again') 
    return cluster_list

def main():
    #INITILIZATION

    #delete and create a new folder for cluster images and trade images
    #turn interactive plotting off, this prevents matplotlib from dispalying all the plots, can still display with plt.show()
    Delete_Create_Folder("Cluster_Images")
    Delete_Create_Folder('Trade_Images')
    plt.ioff()

    if Initialize:
        primary = Initialization()
        primary_dataset = primary.complete_dataset
    else: primary_dataset = np.genfromtxt('Execution_Files\Primary.csv', delimiter=',')

    #CLUSTERING

    if Clustering:
        #cluster and assign labels, plot average clusters and save image to drive
        clusters = Cluster(primary_dataset)
        training_data = clusters.labeled_trimmed
    else: training_data = np.genfromtxt('Execution_Files\Training_Data.csv', delimiter=',')

    #EXECUTION

    #prepare training data
    X_train = training_data[:,0:24]
    y_train = training_data[:,24]

    #create and train KNN classifier
    knn = KNN_Classifier()
    knn.fit(X_train, y_train)

    #take comma delimited input and convert to list of integers
    buy_indicators = Input_and_Convert('Buy Indicators: ')

    #load test datasets: X_test_1, X_test_2, etc. and initiate Main_Loop parameters
    live_counter = 0
    trade_record = []
    Test_1 = Test_Data('X_test_1')
    Test_2 = Test_Data('X_test_2')
    test_array_list = [Test_1, Test_2]

    while live_counter < len(Test_1.array) - 10:
        
        #increment live_counter
        live_counter += 1
        
        for each in test_array_list:
            each.Trade_Loop()

    np.savetxt("Execution_Files\Trading_Record.csv", np.array(trade_record), delimiter=",")
    print('Number of Trades:', len(trade_record))
    print('Ending Account Value:', Account_Value)
    
main()

