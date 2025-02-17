import pandas as pd
import sys

filename = sys.argv[1]
df=pd.read_excel(filename)
df['Polarity'] = df['Polarity'].map({'Negative': 0, 'Positive': 1, 'Neutral': 2})

#Split Train Test
df = df.sample(frac=1, random_state=0).reset_index(drop=True)  
criteria = df['Polarity'] == 1  
posdf = df[criteria]  
criteria = df['Polarity'] == 0  
negdf = df[criteria] 
criteria = df['Polarity'] == 2  
neudf = df[criteria]   
trainpos=posdf.sample(frac=0.7, random_state=0)  
testpos=posdf.drop(trainpos.index)  
trainneg=negdf.sample(frac=0.7, random_state=0)  
testneg=negdf.drop(trainneg.index)  
trainneu=neudf.sample(frac=0.7, random_state=0)  
testneu=neudf.drop(trainneu.index)  
traindata = pd.concat([trainpos, trainneg, trainneu], ignore_index=True)  
testdata = pd.concat([testpos, testneg, testneu], ignore_index=True)

traindata.to_excel("Traindata.xlsx")
testdata.to_excel("Testdata.xlsx")