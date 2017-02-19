def getSuggestionForAuthor(inAuthorId):
    mySuggestions=pd.read_csv(r'C:\Users\Barry\Anaconda\AIHackText\Random\AuthorSuggestions.csv')#.to_dict()
    return dict(zip(mySuggestions.Author, mySuggestions.Suggestions))[inAuthorId]

getSuggestionForAuthor(70535)