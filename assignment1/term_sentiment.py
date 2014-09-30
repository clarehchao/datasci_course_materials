from __future__ import division  #to ensure floating-point division
import sys
import json
import re


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
    tweetscore = []
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
        #print score
        tweetscore.append(score)
    return tweetscore
    
def ScoreNewTerm(fname,thedict,tweetscore):
    allines = open(fname).readlines()
    newtermscore = {}
    extra = ['(', ')', '?', ':', ';', ',', '.', '!', '/', '"', "'","^","+","&","@"]
    for i in range(0,len(allines)):
        tmp = json.loads(allines[i])
        txt = tmp.get(u'text',u'')     # u for unicode
        lang = tmp.get(u'lang',u'')
        if lang == 'en':   #only score English tweets
            #print txt
            thetxt = txt.encode('utf-8')   # convert unicode to str
            thetxtlist = [t.strip() for t in re.findall(r'\b.*?\S.*?(?:\b|$)', thetxt)]
            
            # get the score of the tweet
            thescore = tweetscore[i]
            tmp1 = [x.lower() for x in thetxtlist if x.lower() not in thedict]
            # clean up all the punctuation, hash symbol, and '\x...'
            tmp2 = [x for x in tmp1 if x not in extra]
            #print tmp2
            for newterm in tmp2:
                if newterm in newtermscore:
                    # the newterm dict store the key and update its value list [score, count]
                    newtermscore[newterm][0] += thescore
                    newtermscore[newterm][1] += 1
                else:
                    newtermscore[newterm] = [thescore,1]
    for key in newtermscore:
        print key, newtermscore[key][0]/newtermscore[key][1]

def main():
    sentfname = sys.argv[1]
    tweetfname = sys.argv[2]
    unidict,multidict,alldict=GetScoreDict(sentfname)
    thetweetscore = ScoreTweet(tweetfname,unidict,multidict,alldict)
    ScoreNewTerm(tweetfname,alldict,thetweetscore)

if __name__ == '__main__':
    main()
