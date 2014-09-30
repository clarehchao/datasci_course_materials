import MapReduce
import sys

"""
Word Count Example in the Simple Python MapReduce Framework
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: order_id
    # value: whole record
    key = record[1]
    value = record
    mr.emit_intermediate(key, value)
    #print key,value
      

def reducer(key, list_of_values):
    # key: word
    # value: list of values
    order = list_of_values[0]
    for v in list_of_values[1:]:
        output = order + v
        mr.emit((output))
        

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)
