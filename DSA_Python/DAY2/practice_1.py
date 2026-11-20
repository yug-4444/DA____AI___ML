def gen(n):
    for i in range(n):
        yield i*i

g=gen(4)
a=list(g)
b=sum(g)
print(b)
next(g)