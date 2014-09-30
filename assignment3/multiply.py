import MapReduce
import sys
from numpy import prod

"""
Multiply matrix A and matrix B: matrix multiplication

algorithm provided by STAFF:

[Mapper]
for each element (i,j) of A, emit((i,k), (j, A[i,j])) for k in 1..N
for each element (j,k) of B, emit((i,k), (j, B[j,k])) for i in 1..L

[Reducer]
for each (i,j)
- group the tuple on the first member e.g.
((1,1), (1,1))
((2,3), (2,4))
((3,4), (3,-3))
((4,-2), (4,0))
- multiply the 2nd tuple member:
((1,1), (1,1)) => 1*1 = 1
((2,3), (2,4)) => 3*4 = 12
((3,4), (3,-3)) => 4*-3 = -12
((4,-2), (4,0)) => -2*0 = 0
- sum it all!
1 + 12 + -12 + 0 = 1
- emit a tuple of key member and the product as the result
(i, k, product) => (1, 1, 1) 

"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

# define the size of the matrix
N = 5
L = 5

def mapper(record):
    # key: 
    # value: nucleotides
    mname,row,col,v = record
    if mname =='a':
        for k in range(N):
            key = (row,k)
            value = (col,v)
            mr.emit_intermediate(key,value)       
    if mname == 'b':
        for i in range(L):
            key = (i,col)
            value = (row,v)
            mr.emit_intermediate(key,value) 

def reducer(key, list_of_values):
    tmp = {}
    for v in list_of_values:
        #print key,v[0],v[1]
        thekey,thevalue = v
        if thekey not in tmp:
            tmp.setdefault(thekey, [])
        tmp[thekey].append(thevalue)
    
    prodsum = 0
    for v in tmp:
        if len(tmp[v]) > 1:
            prodsum += prod(tmp[v])
    mr.emit((key[0],key[1],prodsum))

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)
