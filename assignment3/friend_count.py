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
    mr.emit_intermediate(a,1)
      

def reducer(key, list_of_values):
    # key: friend name pair
    # value: count
    count = 0
    for v in list_of_values:
        count += v
    mr.emit((key,count))
        

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)
