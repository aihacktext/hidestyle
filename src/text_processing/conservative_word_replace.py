from nltk.tokenize.moses import MosesDetokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet as wn

detokenizer = MosesDetokenizer()

def replaceAdjectives(inWord,inCat):
    if inCat in ['JJ','RB']:
        
        if len(wn.synsets(inWord))==1:
            mySynset=(wn.synsets(inWord))
            
            if mySynset != []:
                if ('adj' in mySynset[0].lexname()) or True:
                    if inWord != mySynset[0].lemma_names()[0] :
                        print('changing: ' + inWord+ ' to: ' + mySynset[0].lemma_names()[0])
                        return mySynset[0].lemma_names()[0]
       
    return inWord

def ReplaceSomeWords(inSampleText):
    modifiedWords=[ replaceAdjectives(x,y) for (x,y) in nltk.pos_tag(word_tokenize(inSampleText))]
    return detokenizer.detokenize(modifiedWords, return_str=True)





