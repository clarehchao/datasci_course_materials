from __future__ import division  #to ensure floating-point division
import sys
import json
import re

def TermHistogram(fname):
    allines = open(fname).readlines()
    TermHisto = {}
    extra = [":", ";", ",", ".", "!", "/", '"', "'","^","&","@","(",")","?","!","+","*","-","_"]
    for i in range(0,len(allines)):
        tmp = json.loads(allines[i])
        txt = tmp.get(u'text',u'')     # u for unicode
        lang = tmp.get(u'lang',u'')
        if lang == 'en':  #only score English tweets
            # convert unicode to str
            thetxt1 = txt.encode('utf-8')
            #print thetxt1
            
            # remove all ascii code
            thetxt2 = unicode(thetxt1,'ascii','ignore').encode('utf-8')
            #print thetxt2
            
            # split the text into word and punctuation 
            thetxtlist = [t.strip() for t in re.findall(r'\b.*?\S.*?(?:\b|$)', thetxt2)]
            
            # filter out the punctuation and symbols
            thefiltertxt = [x for x in thetxtlist if x not in extra]
            
            # filter out a bit more
            morefiltertxt = [ss for ss in thefiltertxt if any([ss.find(exstr)>=0 for exstr in extra]) == False]
            
            # update the term dictionary with word counts
            for aword in morefiltertxt:
                if aword in TermHisto:
                    TermHisto[aword] += 1
                else:
                    TermHisto[aword] = 1
    
    # compute the frequency
    f = 1./sum(TermHisto.values())
    TermHisto.update((k,v*f) for k,v in TermHisto.items())
    
    #print TermHisto.items()
    
    # print out the term histogram
    print '\n'.join(['%s %s' % (k, ('%('+k+')s') % TermHisto) for k in TermHisto])            
        

def main():
    tweetfname = sys.argv[1]
    TermHistogram(tweetfname)  

if __name__ == '__main__':
    main()
