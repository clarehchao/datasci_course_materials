import sys
import json
import re

def hw():
    print 'Hello, world!'

def lines(fp):
    print str(len(fp.readlines()))

def GetScoreDict(fname):
    # split the dictionary into unigram and multigram
    unigramdict = {}
    multigramdict = {}
    alldict = {}
    ofile = open(fname)
    for line in ofile:
        term,score = line.split('\t')   # the file is tab delimited
        alldict[term] = int(score)
        if len(term.split()) > 1:
            multigramdict[term] = int(score)
        else:
            unigramdict[term] = int(score)
    #print 'Got score dictionary!'
    #print scoredict.items()  #print every pair (key, value) of the dictonary
    return unigramdict,multigramdict,alldict
    
def ScoreTweet(fname,unigramdict,multigramdict,alldict):
    allines = open(fname).readlines()
    for i in range(0,len(allines)):
        tmp = json.loads(allines[i])
        txt = tmp.get(u'text',u'')     # u for unicode
        lang = tmp.get(u'lang',u'')
        if lang == 'en':  #only score English tweets
            #print txt
            thetxt = txt.encode('utf-8')   # convert unicode to str
            # check for words in multigram: improve this code by considering multigram!
 
            # check for words in unigram: the course assignemtn only tests for unigram!
            #thetxtlist = thetxt.split()
            thetxtlist = [t.strip() for t in re.findall(r'\b.*?\S.*?(?:\b|$)', thetxt)]
            #print thetxtlist
            score = 0
            for atxt in thetxtlist:
                latxt = atxt.lower()    # turn string into lower case
                # check the combo text of the neighboring words...!!!!
                if latxt in alldict:
                    score = score + alldict[latxt]
        else:
            score = 0
        print score            
        

def main():
    sentfname = sys.argv[1]
    tweetfname = sys.argv[2]
    #sent_file = open(sentfname)
    #tweet_file = open(tweetfname)
    #hw()
    #lines(sent_file)
    #lines(tweet_file)
    unidict,multidict,alldict = GetScoreDict(sentfname)
    ScoreTweet(tweetfname,unidict,multidict,alldict)
    
    
    

if __name__ == '__main__':
    main()
