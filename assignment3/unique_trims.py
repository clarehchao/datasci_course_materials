import MapReduce
import sys

"""
Word Count Example in the Simple Python MapReduce Framework
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: seq id
    # value: nucleotides
    dna = record[1]
    newdna = dna[:-10]
    mr.emit_intermediate(newdna,1)

def reducer(key, list_of_values):
    # key: shorten dna
    # value: count
    mr.emit(key)

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)
