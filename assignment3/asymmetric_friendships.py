import MapReduce
import sys

"""
Word Count Example in the Simple Python MapReduce Framework
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: friend A
    # value: friend B
    a = record[0]
    b = record[1]
    #print a,b
    mr.emit_intermediate((a,b),1)
    mr.emit_intermediate((b,a),-1)

def reducer(key, list_of_values):
    # key: friend A
    # value: associated friends
    count = 0
    #print key
    for v in list_of_values:
        #print v
        count += v
    if count != 0:
        mr.emit(key)

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)
