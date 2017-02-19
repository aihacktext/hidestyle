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



def getRecommendations(inSampleText, TextStatsOverallMeansReadIn, TextStatsReadIn, interestingColumns):
    StatsForOneReview =getStatsFromReviewText(inSampleText,imbdFileNameDir).drop(['Author','Title']).convert_objects(convert_numeric=True)

    StatsForOneReviewPerc= copy.deepcopy(StatsForOneReview)
    for mycol in StatsForOneReview.index:
        StatsForOneReviewPerc[mycol+'_per']=percentileofscore(TextStatsReadIn[mycol],StatsForOneReview[mycol])/100

    sig = getSignature(StatsForOneReviewPerc[interestingColumns])
    msg =  '\n'.join([getRecommendString(StatsForOneReviewPerc,x,TextStatsOverallMeansReadIn) for x in sig])
    return msg, sig

def analyze_style(SampleText):
    interestingColumns=[u'MeanWordLen', u'MeanSentenceLen', u'MeanParagraphLen', u'Commas',
            u'Semicolons', u'Exclamations', u'Colons', u'Dashes',
            u'Mdashes', u'Ands', u'Buts', u'Howevers', u'Ifs', u'Thats', u'Mores',
            u'Musts', u'Mights', u'This', u'Verys']

    interestingColumns=[x+'_per' for x  in interestingColumns]
    TextStatsOverallMeansReadIn=pd.read_csv(r'../TextStatsOverallMeans.csv')
    TextStatsReadIn=pd.read_csv(r'../TextStats.csv')
    TextStatsOverallMeansReadIn=pd.Series(data= TextStatsOverallMeansReadIn['504335.1'].values,index=TextStatsOverallMeansReadIn['Author'].values)

    getRecommendations(SampleText, TextStatsOverallMeansReadIn, TextStatsReadIn, interestingColumns)
