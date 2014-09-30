import json
import sys
import pprint

fdir = '/home/clareh/Documents/coursera/IntroDataScience/datasci_course_materials/assignment3'
#jfname1 = fdir + '/problem1.json'
#jfname2 = fdir + '/solutions/inverted_index.json'
#jfname2 = fdir + '/solutions/join.json'
#jfname2 = fdir + '/solutions/asymmetric_friendships.json'
#jfname2 = fdir + '/data/friends.json'
#data1 = open(jfname1)
#jfname2 = fdir + '/data/dna.json'
#jfname2 = fdir + '/solutions/unique_trims.json'
#jfname2 = fdir + '/problem5.json'
jfname2 = fdir + '/solutions/multiply.json'
data2 = open(jfname2)
"""
for line in data1:
    oh = json.loads(line)
    print oh
"""
count = 0
for line in data2:
    oh = json.loads(line)
    count +=1
    print oh
print count





