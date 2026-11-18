# Objective: Write a single Python script demonstrating three distinct memory behaviors:
# A function with a mutable default list called 3 times (twice without an argument, once with an explicit list).
# A closure factory outer() returning inner(val) that mutates a nonlocal list across two separate closure instances f1 and f2.
# A list of 4 lambda functions created inside a loop that exhibits late-binding, alongside its fixed early-bound equivalent.
# The Diagnostic: Predict and prove the exact memory IDs (id()) and outputs in your terminal.

#1.
def items(product ,cart=[]):
    cart.append(product)
    print(f"our cart: {id(cart)} contains items {cart}")

items("ice")
items("cola")
items("atta",[])
items("jera",[])
c1=[]
items("icecream",c1)
items("vodka",c1)
items("pepsi")

#2
def outer_factory():
    count_member=0
    record_id=[]
    def add_member():
        nonlocal count_member
        count_member+=1
        record_id.append(count_member)
        print(f"factory_id: {id(record_id)} and the current record is: {record_id}")
        return count_member+id(record_id)
    return add_member

factory_1=outer_factory()
factory_2=outer_factory()
factory_3=outer_factory()
m1=factory_1()
m2=factory_1()
m3=factory_2()
m4=factory_3()
m5=factory_2()
m6=factory_1()
m7=factory_2()

#3
func=[lambda :i for i in range(4)]
func_fix=[lambda curr_i=i:curr_i for i in range(4)]
fc=[f() for f in func]
fcfix=[f() for f in func_fix]

print(f"late binding -> {fc}")
print(f"fix binding -> {fcfix}")

#4
matrix_a=[[0]*3]*4
matrix_b=[[0,0,0] for _ in range(4)]
matrix_a[0][0]=1
matrix_b[0][0]=1
print(f"matrix a: {matrix_a}")
print(f"matrix b: {matrix_b}")