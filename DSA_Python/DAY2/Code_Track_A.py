"""    
Write a single Python diagnostic script covering two behaviours. Predict every output and id() match on paper before executing:
Start with X = [[1, 2], (3, [4, 5])]. Create four variants: direct assignment (=), list slice ([:]), copy.copy(X), and copy.deepcopy(X). 
Then mutate X[0].append(99) and X[1][1].append(88). Verify which variants reflect each mutation and compare the id() of the 
outer list, inner list, and inner tuple across all four.Test t = (1, [2]) under two operations:
Attempting to use (1, 2) vs. (1, [2]) as keys in a Python dict.
Executing t[1] += [3] inside a try ... except TypeError block, then printing t after the except block finishes.
"""
import copy
x=[[1,2],(3,[4,5])]
da=x
ls=x[:]
c=copy.copy(x)
dc=copy.deepcopy(x)
x[0].append(99)
x[1][1].append(88)
print("case-> x vs da")
print(x)
print(da)
print(id(x)==id(da),id(x[0])==id(da[0]),id(x[1])==id(da[1]),id(x[1][0])==id(da[1][0]),id(x[1][1])==id(da[1][1]))
print("case-> x vs ls")
print(x)
print(ls)
print(id(x)==id(ls),id(x[0])==id(ls[0]),id(x[1])==id(ls[1]),id(x[1][0])==id(ls[1][0]),id(x[1][1])==id(ls[1][1]))
print("case-> x vs c")
print(x)
print(c)
print(id(x)==id(c),id(x[0])==id(c[0]),id(x[1])==id(c[1]),id(x[1][0])==id(c[1][0]),id(x[1][1])==id(c[1][1]))
print("case-> x vs dc")
print(x)
print(dc)
print(id(x)==id(dc),id(x[0])==id(dc[0]),id(x[1])==id(dc[1]),id(x[1][0])==id(dc[1][0]),id(x[1][1])==id(dc[1][1]))
#dictionary hashable keys that need to immutable 
good_key=(1,2,3)
good_key_1=1
bad_key=(1,[2,3])
bad_key_1=[1,2,3]
print(hash(good_key),hash(good_key_1))
dic={}
try:
    dic[good_key]="a"
except TypeError as e:
    print(f"there was error {e} in key {good_key}")
try:
    dic[good_key_1]="b"
except TypeError as e:
    print(f"there was error {e} in key {good_key_1}")
try:
    dic[bad_key]="c"
except TypeError as e:
    print(f"there was error {e} in key {bad_key}")
try:
    dic[bad_key_1]="d"
except TypeError as e:
    print(f"there was error {e} in key {bad_key_1}")
print(dic)

try:
    good_key+=1
except TypeError as e:
    print(f"there was error {e} in key {good_key}")
try:
    good_key_1+=1
except TypeError as e:
    print(f"there was error {e} in key {good_key_1}")
try:
    bad_key[1]+=[1]
except TypeError as e:
    print(f"there was error {e} in key {bad_key}")
try:
    bad_key_1+=[1]
except TypeError as e:
    print(f"there was error {e} in key {bad_key_1}")
print(good_key,good_key_1,bad_key,bad_key_1)

#need of copy in list
def trace_me(nums, idx, path, res):
    if idx == len(nums):
        res.append(copy.deepcopy(path))
        return
    path.append(nums[idx])
    trace_me(nums, idx + 1, path, res)
    path.pop()
    trace_me(nums, idx + 1, path, res)

out = []
trace_me([10, 20], 0, [], out)
for i in range(len(out)):
    print(id(out[i]))
print(out)