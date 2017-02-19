import nltk

import copy
import pandas as pd
import numpy as np
from scipy.stats import  percentileofscore
from stylometry.extract import *

def getStatsFromReviewText(inReviewText,inDir):
    #very hacky1 write Review to file so it can be read by stylometry
    myTempFileName=inDir+'\\'+'testWrite.txt'
    myFile = open(myTempFileName,'w+')
    #print(inReviewText)
    myFile.write(inReviewText)
    myFile.close()
    
    #get stats From stylometry
    notDickens=StyloDocument(myTempFileName)
    headers=notDickens.csv_header().split(',')
    myData=notDickens.csv_output().split(',')
    myReturnData= pd.Series(data=myData ,index=headers)
    
    return myReturnData
  
  
def getSignature(inPercentStats):
    HighValue=.86
    LowValue=.14
    signatures=inPercentStats[(inPercentStats>HighValue)|(inPercentStats<LowValue)]
    mySignature= zip(signatures.tolist(),signatures.index.tolist())
    return [['High' if x>.5 else 'Low',y[:-4]] for (x,y) in mySignature]

def getRecommendString(inSeries,HighLowAndName,inAverages):
    FieldName=HighLowAndName[1]
    HighLow=HighLowAndName[0]
    #print(FieldName)
    #print(HighLow)
    myAnswer =FieldName+ ' is ' + HighLow +'. Your average is '+str(np.round(inSeries[FieldName],1)) \
        +' vs. Average of All reviewers : '+str(np.round(inAverages[FieldName],1))  \
        + ' your average Review is in the ' +str(np.round(inSeries[FieldName+'_per']*100,0)) +' Percentile'
        #
    return myAnswer
    
    
interestingColumns=[u'MeanWordLen', u'MeanSentenceLen', u'MeanParagraphLen', u'Commas',
        u'Semicolons', u'Exclamations', u'Colons', u'Dashes',
        u'Mdashes', u'Ands', u'Buts', u'Howevers', u'Ifs', u'Thats', u'Mores',
        u'Musts', u'Mights', u'This', u'Verys']
 
interestingColumns=[x+'_per' for x  in interestingColumns]
TextStatsOverallMeansReadIn=pd.read_csv(r'C:\Users\Barry\Anaconda\AIHackText\Random\TextStatsOverallMeans.csv')
TextStatsReadIn=pd.read_csv(r'C:\Users\Barry\Anaconda\AIHackText\Random\TextStats.csv')
TextStatsOverallMeansReadIn=pd.Series(data= TextStatsOverallMeansReadIn['504335.1'].values,index=TextStatsOverallMeansReadIn['Author'].values)




imbdFileName='imdb62_p0.txt'
imbdFileNameDir=r'C:\Users\Barry\Anaconda\AIHackText\Random'

f=open(imbdFileNameDir+'\\'+imbdFileName)
SampleText=f.readline().split(None,4)[4]


def getRecommendations(inSampleText):
    StatsForOneReview =getStatsFromReviewText(inSampleText,imbdFileNameDir).drop(['Author','Title']).convert_objects(convert_numeric=True)

    StatsForOneReviewPerc= copy.deepcopy(StatsForOneReview)
    for mycol in StatsForOneReview.index:
        StatsForOneReviewPerc[mycol+'_per']=percentileofscore(TextStatsReadIn[mycol],StatsForOneReview[mycol])/100

    return '\n'.join([getRecommendString(StatsForOneReviewPerc,x,TextStatsOverallMeansReadIn) for x in getSignature(StatsForOneReviewPerc[interestingColumns])])
    
    
f=open(imbdFileNameDir+'\\'+imbdFileName)
f.readline().split(None,4)[4]
f.readline().split(None,4)[4]
f.readline().split(None,4)[4]
f.readline().split(None,4)[4]
f.readline().split(None,4)[4]
                           
SampleText=f.readline().split(None,4)[4]


getRecommendations(SampleText)
