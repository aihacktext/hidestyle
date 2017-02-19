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
    myReturnData['ReviewScore
    


##get Stats on the first 10 reviews in the data

imbdFileName='imdb62_p0.txt'
imbdFileNameDir=r'C:\Users\Barry\Anaconda\AIHackText\Random'

f=open(imbdFileNameDir+'\\'+imbdFileName)
myReviewStats2=[getStatsFromReview(x.split(None,4),imbdFileNameDir) for x in f.readlines()[:10]]

myReviewStatsFrame=pd.concat(myReviewStats2,axis=1).T.drop('Title',axis=1).convert_objects(convert_numeric=True)
myReviewStatsFrame