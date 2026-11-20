# 1. RangeClone (Custom Iterable & Iterator Protocol)Base Statement: Implement a pure Python class RangeClone(start, stop, step=1) 
# replicating Python's built-in range object without storing elements in a list or calling range().Required Capabilities:Support
#  both positive and negative step (raise ValueError if step == 0).Support len(r), indexing r[k] 
# (including negative indices and IndexError on out-of-bounds), membership test x in r, and for x in r.
# Must be re-iterable (running two consecutive for loops over the same RangeClone instance must both yield the full sequence).
# Strict Constraints: __len__, __getitem__, and __contains__ must all run in strict $\mathcal{O}(1)$ Time and 
# $\mathcal{O}(1)$ Auxiliary Space (zero loops).

# 1 :
"""
RangeClone
│
├── __len__()
├── __getitem__()
├── __contains__()
├── __iter__()
│
└── RangeIterator
        ├── __iter__()
        └── __next__()
"""
class RangeIterator:
    def __init__(self,range_obj):
        self.index=0
        self.range_obj=range_obj
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index>=len(self.range_obj):
            raise StopIteration("iteration reached end")
        value=self.range_obj.start+self.index*self.range_obj.step
        self.index+=1
        return value

class RangeClone:
    def __init__(self,start,stop=None,step=1):
        if step==0:
            raise ValueError("Step size 0 means that out start don't move")
        if stop is None:
            start,stop=0,start
        if step>0:
            if start>=stop:
                self.length=0
            else:
                self.length=(stop-start+step-1)//step
        else:
            step_abs=-step
            if start<=stop:
                self.length=0
            else:
                self.length=(start-stop+step_abs-1)//step_abs
        self.start=start
        self.stop=stop
        self.step=step

    def __getitem__(self, key):
        if key<0:
            key+=self.length
        if key<0 or key>=self.length:
            raise IndexError("outof bound index used")
        return self.start+key*self.step
    
    def __len__(self):
        return self.length
    
    def __contains__(self, item):
        if self.length==0:
            return False
        if self.step>0:
            return(self.start<=item<self.stop and (item-self.start)%self.step==0)
        else:
            return(self.stop<item<=self.start and (self.start-item)%(-self.step)==0)
        
    def __iter__(self):
        return RangeIterator(self)

# 2. manual_deepcopy(obj) (Recursive Memory Graph Cloner)Base Statement: Write a pure Python recursive function manual_deepcopy(obj)
# from scratch (no copy module) that deep-copies arbitrarily nested list, tuple, dict, and set structures.Strict Constraints:
# Must preserve structure and terminate cleanly on self-referential / cyclic structures (e.g., a = [1, 2]; a.append(a)) without
# raising RecursionError. Must run in $\mathcal{O}(V + E)$ Time where $V$ is total container objects and $E$ is total references.

def manual_deepcopy(obj,memo=None):
    if memo is None:
        memo={}
    #in dict key must be immutable so list ,sets ,tuple(with mutable obj ) are not allowed to be keys so lets store there id that is address which is unique and work as key
    obj_id=id(obj)
    if isinstance(obj,(int,float,bool,str,type(None))):
        return obj

    if obj_id in memo.keys():
        return memo[obj_id]
    
    if isinstance(obj,list):
        clone=[]
        memo[obj_id]=clone
        for item in obj:
            clone.append(manual_deepcopy(item,memo))
        return clone
    
    if isinstance(obj,set):
        clone=set()
        memo[obj_id]=clone
        for item in obj:
            clone.add(manual_deepcopy(item,memo))
        return clone

    if isinstance(obj,dict):
        clone={}
        memo[obj_id]=clone
        for k,v in obj.items():
            clone[manual_deepcopy(k,memo)]=manual_deepcopy(v,memo)
        return clone

    if isinstance(obj,tuple):
        clone=None
        memo[obj_id]=clone
        clone=tuple(manual_deepcopy(item,memo) for item in obj)
        memo[obj_id]=clone
        return clone
    return obj

#but this don't work tuple cycles 
#lets use the better 4 Phase version
#work everything except tuple<->tuple cycle
class tuple_obj:
    def __init__(self):
        self.list_item=[]

def clone_corpse(obj,memo=None):
    if memo is None:
        memo={}
    obj_id=id(obj)
    if obj_id in memo.keys():
        return
    if isinstance(obj,(int,float,bool,str,type(None))):
        return
    elif isinstance(obj,list):
        memo[obj_id]=[]
        for item in obj:
            clone_corpse(item,memo)
    elif isinstance(obj,set):
        memo[obj_id]=set()
        for item in obj:
            clone_corpse(item,memo)
    elif isinstance(obj,dict):
        memo[obj_id]={}
        for k,v in obj.items():
            clone_corpse(k,memo)
            clone_corpse(v,memo)
    elif isinstance(obj,tuple):
        memo[obj_id]=tuple_obj()
        for item in obj:
            clone_corpse(item,memo)

def illusion_connect(obj,memo,visited=None):#tuple obj are not real that are suppose to be connected 
    if visited is None:
        visited=set()
    obj_id=id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)
    if isinstance(obj,(int,float,str,bool,type(None))):
        return
    elif isinstance(obj,list):
        clone=memo[obj_id]
        for item in obj:
            clone.append(memo.get(id(item),item))
            illusion_connect(item,memo,visited)
    elif isinstance(obj,set):
        clone=memo[obj_id]
        for item in obj:
            clone.add(memo.get(id(item),item))
            illusion_connect(item,memo,visited)
    elif isinstance(obj,dict):
        clone=memo[obj_id]
        for k,v in obj.items():
            clone[memo.get(id(k),k)]=memo.get(id(v),v)
            illusion_connect(k,memo,visited)
            illusion_connect(v,memo,visited)
    elif isinstance(obj,tuple):
        clone=memo[obj_id]
        for item in obj:
            clone.list_item.append(memo.get(id(item),item))
            illusion_connect(item,memo,visited)

def find_tuple_obj(obj,memo,placeholder_tuple=None,visited=None):
    if placeholder_tuple is None:
        placeholder_tuple={}
    if visited is None:
        visited=set()
    obj_id=id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)
    if isinstance(obj,(int,float,str,bool,type(None))):
        return
    if isinstance(obj,(list,set)):
        for items in obj:
            find_tuple_obj(items,memo,placeholder_tuple,visited)
    elif isinstance(obj,dict):
        for k,v in obj.items():
            find_tuple_obj(k,memo,placeholder_tuple,visited)
            find_tuple_obj(v,memo,placeholder_tuple,visited)
    elif isinstance(obj,tuple):
        for item in obj:
            find_tuple_obj(item,memo,placeholder_tuple,visited)
        placehold=memo[obj_id]
        temp = []
        for x in placehold.list_item:
            if isinstance(x, tuple_obj):
                temp.append(placeholder_tuple.get(x, x))
            else:
                temp.append(x)
        true_tuple = tuple(temp)
        placeholder_tuple[placehold] = true_tuple
        memo[obj_id]=true_tuple

def illusion_to_reality(obj,memo,placeholder_tuple=None,visited=None):
    if placeholder_tuple is None:
        placeholder_tuple={}
    if visited is None:
        visited=set()
    if isinstance(obj,(int,float,str,bool,type(None))):
        return
    obj_id=id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)
    clone=memo[id(obj)]
    if isinstance(clone,list):
        for i in range(len(clone)):
            if isinstance(clone[i],tuple_obj):
                clone[i]=(placeholder_tuple[clone[i]])
        for item in obj:
            illusion_to_reality(item,memo,placeholder_tuple,visited)
    elif isinstance(clone,set):
        replace=set()
        for item in clone:
            if isinstance(item,tuple_obj):
                item=placeholder_tuple[item]
            replace.add(item)
        clone.clear()
        clone.update(replace)
        for item in obj:
            illusion_to_reality(item,memo,placeholder_tuple,visited)
    elif isinstance(clone,dict):
        replace={}
        for k,v in clone.items():
            if isinstance(k,tuple_obj):
                k=(placeholder_tuple[k])
            if isinstance(v,tuple_obj):
                v=(placeholder_tuple[v])
            replace[k]=v
        clone.clear()
        clone.update(replace)
        for k,v in obj.items():
            illusion_to_reality(k,memo,placeholder_tuple,visited)
            illusion_to_reality(v,memo,placeholder_tuple,visited)

def manual_deepcopy(obj):
    memo={}
    clone_corpse(obj,memo)
    illusion_connect(obj,memo)
    placeholder_tuple={}
    find_tuple_obj(obj,memo,placeholder_tuple)
    illusion_to_reality(obj,memo,placeholder_tuple)
    return memo.get(id(obj),obj)