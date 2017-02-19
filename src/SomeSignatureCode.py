import nltk

import copy
from stylometry.extract import *

def getStatsFromReview(inReview,inDir):
    #very hacky1 write Review to file so it can be read by stylometry

    myTempFileName=inDir+'\\'+'testWrite.txt'
    myFile = open(myTempFileName,'w+')
    myFile.write(inReview[4])
    myFile.close()
    
    #get stats From stylometry
    notDickens=StyloDocument(myTempFileName)
    headers=notDickens.csv_header().split(',')
    myData=notDickens.csv_output().split(',')
    myReturnData= pd.Series(data=myData ,index=headers)
        
    ##AddExtra Data
    myReturnData['Author']=inReview[1]
    myReturnData['ReviewId']=inReview[0]
    myReturnData['ReviewId2']=inReview[2]
    myReturnData['ReviewScore']=inReview[3]
    return myReturnData



imbdFileName='imdb62_p0.txt'
imbdFileNameDir=r'C:\Users\Barry\Anaconda\AIHackText\Random'

f=open(imbdFileNameDir+'\\'+imbdFileName)
myReviewStats=[getStatsFromReview(x.split(None,4),imbdFileNameDir) for x in f.readlines()]

TextStats=pd.concat(myReviewStats,axis=1).T.drop('Title',axis=1).convert_objects(convert_numeric=True)


TextStatsPerc= copy.deepcopy(TextStats)
for mycol in TextStats.columns:
    TextStatsPerc[mycol+'_per']=TextStats[mycol].rank()/len(TextStats)
    
interestingColumns=[u'MeanWordLen', u'MeanSentenceLen', u'MeanParagraphLen', u'Commas',
       u'Semicolons', u'Exclamations', u'Colons', u'Dashes',
       u'Mdashes', u'Ands', u'Buts', u'Howevers', u'Ifs', u'Thats', u'Mores',
       u'Musts', u'Mights', u'This', u'Verys']

interestingColumns=[x+'_per' for x  in interestingColumns]


def getSignature(inPercentStats):
    HighValue=.8
    LowValue=.2
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

TextStatsOverallMeans=TextStatsPerc.mean()

TextStatsPercMean['Signatures']=TextStatsPercMean[interestingColumns].apply(lambda x:getSignature(x),axis=1)
TextStatsPercMean['Suggestions']=TextStatsPercMean.apply(lambda x:[getRecommendString(x,y,TextStatsOverallMeans) for y in x.Signatures] if x.Signatures <>[] else '' ,axis=1)
TextStatsPercMean['Suggestions']=TextStatsPercMean.Suggestions.apply(lambda x:'\n'.join(x))


TextStatsPercMean.reset_index()[['Author','Suggestions']].to_csv(r'C:\Users\Barry\Anaconda\AIHackText\Random\AuthorSuggestions.csv',index=False)