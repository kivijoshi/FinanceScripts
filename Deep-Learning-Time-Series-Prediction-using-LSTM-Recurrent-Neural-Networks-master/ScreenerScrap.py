import json
from pandas.io.json import json_normalize
import pandas as pd
import numpy as np
import os
NumberOfPages = 3

# https://www.screener.in/api/screens/92716/?limit=100&page=1


PandasData = pd.DataFrame()
for page in range(NumberOfPages):
    FileName = os.path.join(os.getcwd(),'Pages',str(page+1)+'.json')
    with open(FileName) as json_data:
        data = json.load(json_data)
        print(data)
        
        CompanyData = []
        for result in data['page']['results']:
            CompanyData.append(result)
            
            
        B = data['page']['ratios']
        B.insert(0,['Name'])
        B.insert(0,['S No'])
        
        columns = []
        for ele in B:
            columns.append(ele[0])
        
        if(page == 0):
            PandasData=pd.DataFrame(CompanyData,columns = columns)
        else:
            TempData=pd.DataFrame(CompanyData,columns = columns)
            PandasData = PandasData.append(TempData)
            
        PandasData['S No'] = np.arange(len(PandasData))
        

FinalData = PandasData


SortedOnROIC = FinalData.sort(['Return on invested capital'],ascending=[0])

SortedOnROIC['ROIC Rank'] = np.arange(len(SortedOnROIC))

SortedOnYield = SortedOnROIC.sort(['Earnings yield'],ascending=[0])

SortedOnYield['Yield Rank'] = np.arange(len(SortedOnYield))

MagicFormulizedData = SortedOnYield
MagicFormulizedData['MagicID'] = MagicFormulizedData['Yield Rank'] + MagicFormulizedData['ROIC Rank']

MagicFormulizedData = MagicFormulizedData.sort(['MagicID','Piotroski score'],ascending=[1,0])

