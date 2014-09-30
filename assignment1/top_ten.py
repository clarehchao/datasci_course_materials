from __future__ import division  #to ensure floating-point division
import sys
import json
import re

def UpdateHisto(thehisto,k):
    if k not in thehisto:
        thehisto[k] = 1
    else:
        thehisto[k] += 1

def TopTenFreq(thehisto):
    thesortkey = sorted(thehisto,key=thehisto.get,reverse=True)
    thesortval = [thehisto[k] for k in thesortkey]
    toptendict = {}
    for i in range(10):
        toptendict[thesortkey[i]] = thesortval[i]
    return toptendict
    
def PrintDict(thedict):
    print '\n'.join(['%s %s' % (k, ('%('+k+')s') % thedict) for k in thedict])        
    

def HashHistogram(fname):
    allines = open(fname).readlines()
    HashtagHisto = {}
    for i in range(0,len(allines)):
        tmp = json.loads(allines[i])
        ent = tmp.get(u'entities',u'')
        lang = tmp.get(u'lang',u'')
        if lang == 'en':
            if ent:
                hashlist = ent[u'hashtags']
                if hashlist:
                    thehash = hashlist[0][u'text']
                    UpdateHisto(HashtagHisto,thehash)
    return HashtagHisto 
               
   

def main():
    tweetfname = sys.argv[1]
    thehisto = HashHistogram(tweetfname)
    topten = TopTenFreq(thehisto)
    PrintDict(topten)

if __name__ == '__main__':
    main()
