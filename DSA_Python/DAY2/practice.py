#copy / deep copy working in memory
import copy
a=[1,(1,[1,2]),[1,2,3]]
print("case1")
b=a
print(id(a)==id(b))
print("case2")
s=a[:]
print(id(a)==id(s))
for _ in range(len(a)):
    print(f"     {_} ->",id(a[_])==id(s[_]))
print("case3")
c=copy.copy(a)
print(id(a)==id(c))
for _ in range(len(a)):
    print(f"     {_} ->",id(a[_])==id(c[_]))
c[1][1].append(3)
c[2].append(4)
a[0]=0
c[2][0]=0
print(c)
print(a)
print("case4")
dc=copy.deepcopy(a)
print(id(a)==id(dc))
for _ in range(len(a)):
    print(f"     {_} ->",id(a[_])==id(dc[_]))
print(id(a[1][1])==id(dc[1][1]))
dc[1][1].append(4)
dc[2].append(5)
a[0]=2
dc[2][0]=6
print(dc)
print(a)

d=(1,{1})
d[1].add(2)
print(d)

l=[1,2,3]
t=(1,2,3)
l[1]+=1
print(l)

print(l[1])
print(l.__getitem__(1))
print(l.__len__())
print(l.__contains__(3))
print(l.__contains__(6))

nums=[1,2,3]
it=iter(nums)
it1=iter(nums)
print(it,it1)
next(it)
next(it)
print(it,*it,it1,*it1)

g=(x for x in range(5))
a=iter(g)
print(g)
next(g)
next(a)
next(g)
print(g,*g,a,*a)

d={}
d[False]=10
d[0]=20
d[0.0]=60
d[a==b]=90
print(next(iter(d)))