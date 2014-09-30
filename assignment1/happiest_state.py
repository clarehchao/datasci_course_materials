from __future__ import division  #to ensure floating-point division
import sys
import json
import re

states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

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

def FindState(thelist):
    tmp = [s for s in thelist if any([states.has_key(s.upper()), s.title() in states.values()])]
    if tmp:
        if states.has_key(tmp[0].upper()):
            return tmp[0]
        elif tmp[0].title() in states.values():
            return [k for k,v in states.items() if v == tmp[0].title()][0] 

    
def AssignStateScore(thescoredict,thestate,score):
    if thestate not in thescoredict:
        thescoredict[thestate] = [score,1]
    else:
        thescoredict[thestate][0] += score
        thescoredict[thestate][1] += 1
        
    
def ScoreStateTweet(fname,unigramdict,multigramdict,alldict):
    allines = open(fname).readlines()
    tweetscore = {}
    for i in range(0,len(allines)):
        tmp = json.loads(allines[i])
        txt = tmp.get(u'text',u'')     # u for unicode
        lang = tmp.get(u'lang',u'')
        if lang == 'en':  #only score English tweets
            # get tweet score
            thetxt = txt.encode('utf-8')   # convert unicode to str
            thetxtlist = [t.strip() for t in re.findall(r'\b.*?\S.*?(?:\b|$)', thetxt)]
            score = 0
            for atxt in thetxtlist:
                latxt = atxt.lower()    # turn string into lower case
                # check the combo text of the neighboring words...!!!!
                if latxt in alldict:
                    score = score + alldict[latxt]
                    
            # get user info
            user = tmp.get(u'user',u'')
            if user:
                userlocation = user[u'location'].encode('utf-8')
                ullist = userlocation.split()
                thestate = FindState(ullist)
                #print thestate
            
            if thestate:
                AssignStateScore(tweetscore,thestate,score) # convert list to str
            else:
                # get place information
                place = tmp.get(u'place',u'')
                if place:
                    location = place[u'full_name'].encode('utf-8')
                    loclist = location.split(',')
                    thestate = FindState(loclist)
                if thestate:
                    AssignStateScore(tweetscore,thestate,score)
            
            
            # get coord information
            #coord = tmp.get(u'coordinates',u'')
            #if coord:
                #print coord[u'coordinates']
       
    # get average tweet score for each state
    avgstatescore = {}
    for k in tweetscore:
        avgstatescore[k] = tweetscore[k][0]/tweetscore[k][1]
        #print k,avgstatescore[k]
    print max(avgstatescore,key=avgstatescore.get)
    
            
           
    
def main():
    sentfname = sys.argv[1]
    tweetfname = sys.argv[2]
    unidict,multidict,alldict=GetScoreDict(sentfname)
    ScoreStateTweet(tweetfname,unidict,multidict,alldict)

if __name__ == '__main__':
    main()
 
    

