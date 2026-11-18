# Topic: Dynamic Array (CPython list Clone)Base Statement: Implement a custom DynamicArray class in Python using
# fixed-size raw array allocations ([None] * capacity). It must track exact resize operations and amortized cost when doubling
# capacity on overflow, and halving capacity at 25% occupancy.Strict Constraints: You are physically forbidden from using 
# .append() or .pop(). You must log the exact element copy counts after $N$ insertions.
class dynamic_array():
    def __init__(self):
        self.array=[None]
        self.capacity=1
        self.current=0
        self.copy_resize=0

    def resize(self):
        self.capacity=self.capacity*2
        copy_array=[None]*self.capacity
        for i in range(self.current):
            copy_array[i]=self.array[i]
            self.copy_resize+=1
        self.array=copy_array

    def shrinkage(self):
        self.capacity=max(1,self.capacity//2)
        copy_array=[None]*self.capacity
        for i in range(self.current):
            copy_array[i]=self.array[i]
            self.copy_resize+=1
        self.array=copy_array

    def append(self,x=0):
        if(self.current==self.capacity):
            self.resize()
        self.array[self.current]=x
        self.current+=1

    def pop(self):
        if(self.current==0):
            raise IndexError("Array empty")
        pr=self.array[self.current-1]
        self.current-=1
        self.array[self.current]=None
        if self.capacity>1 and 4*self.current<=self.capacity:
            self.shrinkage()
        return pr

    def display_and_cost(self):
        print(self.array[:self.current])
        print(f"size = {self.current}")
        print(f"capacity = {self.capacity}")
        print(f"copy cost = {self.copy_resize}")

    def give_back(self):
        return self.array,self.current


arr = dynamic_array()

for i in range(10):
    arr.append(i)
arr.display_and_cost()
for i in range(8):
    arr.pop()
arr.display_and_cost()
for i in range(10):
    arr.append(i+10)
arr.display_and_cost()
arr