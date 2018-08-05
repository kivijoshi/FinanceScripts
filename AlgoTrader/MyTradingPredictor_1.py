
# coding: utf-8

# In[111]:

import pandas as pd
from scipy import optimize
data = pd.read_csv('C:\\Users\\kaust\\Downloads\\WinPython-64bit-3.5.2.3Qt5\\myScripts\\MCX_monthlyData.txt')
data = data.drop(['DATE'],axis=1)
data.head()


# In[112]:

def Model(*params):
    print(params[0][0])
    print(params)
    window = params[0][0]
    window_2 = params[0][1]
    window_3 = params[0][2]
    windowEnter = params[0][3] 
    EnterDifference = params[0][4]
    MovingHigh = []
    MovingLow = []
    MovingAvarage = []
    MovingAvarage_2 = []
    MovingAvarage_3 = []
    #window = 400
    #window_2 = 20
    #window_3 = 120
    #windowEnter = 15
    #EnterDifference = 13
    Enter = []
    # Up = 1 down = 0
    for index, row in data.iterrows():
       
        
        if (index+windowEnter <= data.shape[0]):
            EnterDiff = [x - row['OPEN'] for x in data['OPEN'][index+1:index+windowEnter]] 
            if(max(EnterDiff)>=EnterDifference):
                Enter.append(1)
            else:
                Enter.append(0)
        else:
            Enter.append(0)
                
        if(index < window):
            MovingAvarage.append(99999999)
        else:
            MovingAvarage.append(data['OPEN'][(index-window):index].mean())
            
        if(index < window_3):
            MovingAvarage_3.append(99999999)
        else:
            MovingAvarage_3.append(data['OPEN'][(index-window_3):index].mean())
            
        if(index < window_2):
            MovingAvarage_2.append(99999999)
            MovingHigh.append(99999999)
            MovingLow.append(99999999)
        else:
            MovingHigh.append(data['HIGH'][(index-window):index].max())
            MovingLow.append(data['LOW'][(index-window):index].max())
            MovingAvarage_2.append(data['OPEN'][(index-window_2):index].mean())
            
    data['Enter'] = Enter
    data['MovingHigh'] = MovingHigh
    data['MovingAvarage'] = MovingAvarage
    data['MovingAvarage_2'] = MovingAvarage_2
    data['MovingAvarage_3'] = MovingAvarage_3
    data['MovingLow'] = MovingLow
    data.head()
    
    
    # In[113]:
    
#    get_ipython().magic('matplotlib inline')
#    
#    import matplotlib.pyplot as plt
#    import seaborn as sb
#    sb.set_style("darkgrid")
    
    #plt.plot(data.index.values[1:100],data['Enter'][1:100],data.index.values[1:100],data['OPEN'][1:100])
    #plt.show()
    
    
    # In[114]:
    
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    #import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression
    from sklearn import svm
    
    
    # In[115]:
    
    data_copy = data[window:]
    data_copy = data_copy.drop(['CLOSE','HIGH','LOW'],axis=1)
    data_copy.head()
    
    
    # In[116]:
    
    y = data_copy['Enter'].values
    data_copy = data_copy.drop(['Enter'],axis=1)
    X = data_copy.values
    data_copy.head()
    
    
    # In[117]:
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50)
    print(Xtrain.shape)
    print(Xtest.shape)
    
    
    # In[118]:
    
    gadaboost = GradientBoostingClassifier()
    gadaboost.fit(Xtrain, ytrain)
    y_val_l = gadaboost.predict_proba(Xtest)
    print("Validation accuracy: ", sum(pd.DataFrame(y_val_l).idxmax(axis=1).values
                                       == ytest)/len(ytest))
    
    
    # In[119]:
    
#    import pickle
#    with open('C:\\Users\\kaust\\Downloads\\WinPython-64bit-3.5.2.3Qt5\\myScripts\\my_dumped_classifier.pkl', 'wb') as fid:
#        pickle.dump(gadaboost, fid)    
#    
    
    # In[120]:
    
    #target = open('C:\\Users\\kaust\\Downloads\\WinPython-64bit-3.5.2.3Qt5\\myScripts\\out.txt', 'w')
    y_pre = pd.DataFrame(y_val_l).idxmax(axis=1).values
    
    Total_Trades = 0
    Total_Matched_Trades = 0
    Total_False_Positives = 0
    Total_False_Negatives = 0
    
    for i in range(Xtest.shape[0]):
        line = str(ytest[i])+','+str(y_pre[i])
        #target.write(line)
        #target.write('\n')
        if(ytest[i] == 1):
           Total_Trades = Total_Trades + 1
        if(ytest[i] == 1 and y_pre[i] == 1):
            Total_Matched_Trades = Total_Matched_Trades + 1
        if(ytest[i] == 0 and y_pre[i] == 1):
            Total_False_Positives = Total_False_Positives + 1
        if(ytest[i] == 1 and y_pre[i] == 0):
            Total_False_Negatives = Total_False_Negatives + 1
    #target.close()
    
    print("Total Trades = "+str(Total_Trades))
    print("Total Matched Trades = "+str(Total_Matched_Trades))
    print("Total False Positive Trades = "+str(Total_False_Positives))
    print("Total False Negative Trades = "+str(Total_False_Negatives))
    return (Total_Trades - Total_Matched_Trades)
    
    # In[121]:
    
#    import numpy as np
#    indices = np.argsort(gadaboost.feature_importances_)[::-1]
#    
#    # Print the feature ranking
#    print('Feature ranking:')
#    
#    for f in range(data_copy.shape[1]):
#        print('%d. feature %d %s (%f)' % (f+1 , indices[f], data_copy.columns[indices[f]],
#                                          gadaboost.feature_importances_[indices[f]]))
    
    
    # In[ ]:
 #window = 400
    #window_2 = 20
    #window_3 = 120
    #windowEnter = 15
    #EnterDifference = 13
params = (slice(20,501,1),slice(20,201,1),slice(20,51,1),slice(10,21,1),slice(10,16,1))
resbrute = optimize.brute(Model, params,full_output=True,finish=optimize.fmin)

