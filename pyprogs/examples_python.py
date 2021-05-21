#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 08:53:22 2021

@author: elham
"""
import numpy as np
from numpy import random
import pandas as pd



class DicExamples:
    def __init__(self):
        self.my_dict = {'data1':100,'data2':-54,'data3':247}
        self.d ={'a':2,'b':3}

    # this example sum values of dictionary
    def example_01(self):
        total = 0
        total = sum(list(self.my_dict.values()))
        print(total)
       
    #take dictionary and sort it by values
    def example_2(self):
        x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
        print("original dictionary:", x)
        dict_d = {k: v for k, v in sorted(x.items(), key=lambda item: item[1],reverse=True)}
        print("Dictionary in descending order by value :", dict_d)
        dict_a={k:v for k,v in sorted(x.items(), key = lambda  item: item[1])}
        print("Dictionary in ascending order by value :", dict_a)
        
    #python script to add a key to a dictionary
    def example_3(self):
        d = {1:2,3:4}
        print(d)
        d.update({4:5})
        print(d)
        
    #concatenate following dictionaries to create a new one.
    def example_4(self):
        dic1={1:10, 2:20}
        dic2={3:30, 4:40}
        dic3={5:50,6:60}
        dict4={}
        for d in (dic1,dic2,dic3): 
            dict4.update(d)
    
        print(dict4)
        
    #present key in dictionary
    def example_5(self):
        d = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60}
        def is_key_present(x):
            if x in d:
                print("Key is present in the dictionary")
            else:
                print("Key is not present in the dictionary")
            
        is_key_present(2)
        is_key_present(50)   
        
    #print keys and values in dictionary
    def example_6(self):
        d = {'x': 10, 'y': 20, 'z': 30} 
        for dic_key, dic_value in d.items():
            print(dic_key,"-->",dic_value)
    
    #generate and print a dictionary that contains a number (between 1 and n) in the form (x, x*x)
    def example_7():
        n = int(input("input number:"))
        #def dic_creator(n):
        dic={}
        for i in range(1,n+1):
            dic[i]=i**2
            
        print(dic)   
        
        print("dictionary between 1,16 in form of x,x**2:")
        d={}
        for i in range(1,16):
            d[i]=i**2
        print(d)
    
    #merge two dictioneries
    def example_8(self):
        d1={"a":2}
        d2={"b":3}
        d3={}
        for i in (d1,d2):
            d3.update(i)
        print(d3)
    
    # print keys and values in dictionary   
    def example_9(self):
        #d = {'a': 2, 'b': 3}
        for i,j in self.d.items():
            print(i,"corresponds to",self.d[i])
    #multiply all items in dictionary
    def example_10(self):
        #d = {'a':2,'b':3}
        mul = 1
        for i , j in self.d.items():
            mul *= self.d[i]
        print(mul)
    
    #remove a key from a dictionary.
    def example_11(self):
        #d ={'a':2,'b':3}
        if 'a' in self.d:
            del self.d['a']
        print(self.d)
        
    #map two lists into a dictionary.
    def example_12(self):
        keys = ['a','b']
        values =[1,2]
        d = dict(zip(keys,values))
        print(d)
    
    #sort a dictionary by key
    def example_13(self):
        d ={1:2,3:4}
        a={k:v for k,v in sorted(d.items(), key=lambda item:item[0])}
        #{k: v for k, v in sorted(x.items(), key=lambda item: item[1],reverse=True)}
        print("%s :%s"%(a.keys(),a.values()))
        
        color_dict = {'red':'#FF0000',
              'green':'#008000',
              'black':'#000000',
              'white':'#FFFFFF'}
        print(sorted(color_dict))
        for key in sorted(color_dict):
            print("%s :%s"%(key,color_dict[key]))
    #get first and last item in dictionary 
    def example_14(self):
        #d ={'z':2,'b':4}
        d_a ={k:v for k,v in sorted(self.d.items(),key=lambda item:item[1])}
        dl = list(d_a.items())
        print(dl)
        print("last is:",dl[-1])
        print("first is:",dl[0])
    #get max and min value in dictinary
        
        key_max = max(self.d.keys(), key = (lambda k:self.d[k]))
        key_min = min(self.d.keys(),key=(lambda k:self.d[k]))
        
        print("max is",self.d[key_max])
        print("min is",self.d[key_min])
        ##alternative solution
        all_values = self.d.values()
        max_value = max(all_values)
        min_value = min(all_values)
        print(max_value)
        print(min_value)
        

class TupleExamples:
    def __init__(self):
        self.test_tuple = (1,5)
    def example_01(self):
        print('This is example tuple 1',self.test_tuple)
    
    def example_02(self):
        print('This is example tuple 2',self.test_tuple)
        

class ListExamples:
    def example_01(self):
        print('This is example list 1')

class MapExamples:
    def __init__(self):
        self.num1 = (1,2,3)
        self.num2 = (4,5,6)
        
    #use map to triple all numbers of a given list of integers  
    def example_01(self):
        number =[1,2,3,4,5]
        result = map(lambda x : 3*x,number)
        print("given number is :",number)
        print("map to trible is :" ,list(result))
        
    def example_02(self):
        z = map(lambda x,y : x+y, self.num1,self.num2)
        print("original;", self.num1, self.num2)
        print("map", list(z))
    # listify the list of string using map
    def example_03(self):
        str1 =["hello","my","name","is",'elham']
        z = map(list, str1) 
        print("original list",str1)
        print("listify ", list(z))
    
    def example_04(self):
        listin = []
        n = int(input ("your number of element") )
        a= list(map(int, input("give your number:").strip().split()))[:n]
        z = map(lambda x:x**2,a)   
        print("list of your number",a)
        print("list of power 2 ",list(z))
    # convert all the characters in uppercase and lowercase and eliminate duplicate letters from a given sequence    
    def example_05(self):
        chrars = {'a', 'b', 'E', 'f', 'a', 'i', 'o', 'U', 'a'}
        def change_char(item):
            return str(item).upper(), str(item).lower()
           
        z = map(change_char , chrars)
        print(set(z))
        
    def example_06(self):
        def printnumber(n):
            i = 0
            while i < n:
                i += 1
                if i == 3:
                    continue
                print(i)
        #z = map(printnumber(6),6)
        print(printnumber(6))

class NumPyExamples:
    def __init__(self):
        self.arr = [1,2,3,4,5]
        
    def example_01(self):
        #check version of numpy
        print("version numpy")
        print(np.__version__)
        #arr1 = np.array(self.arr)
        print("create a 2 dimension array")
        arr2 = np.array([[1,2,3],[4,5,6]])
        print(arr2)
        print("check type array")
        print(type(arr2))
        print("check dimension of array")
        print(arr2.ndim) # dimension array
        print("slicing of array")
        print(arr2[0:2,0:2])#slicing aray
        print("creat random array")
        x = random.rand(5)
        print(x)
        print("create random array with int numbers")
        z = random.randint(100,size=5,dtype=int)
        print(z)
        print("create an array")
        arr3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        print(arr3)
        print("shape of array")
        print(arr3.shape)
        print("reshape array (4,2)")
        print(arr3.reshape(4,2))
        
class PanDasExamples:
    def __init__(self):
        self.x = 5
    def example_01(self):
        print("check version pandas")
        print(pd.__version__)
        mydataset = { 'cars': ["BMW", "Volvo", "Ford"],
                     'passings': [3, 7, 2]}
        myvar=pd.DataFrame(mydataset)
        print(myvar)
        print("pandas series")
        a = [1,2,3]
        print(pd.Series(a))
        print("create a pandas series from dictionart")
        calories = {"day1": 300, "day2":400, "day3":500}
        print("my calories",calories)
        print(pd.Series(calories))
        data ={"calories": [300,400,500],"duration": [3,4,5]}
        print("create data frame form two series",data)
        print(pd.DataFrame(data,index = ["day1","day2","day3"]))
        df = pd.DataFrame(data)
        print(df.loc[0])
        print(df.loc[[0,1]])
        print(df.corr())
        print(df.dropna())
        #replace nan with value 
        print(df.fillna(130))
        print(df["calories"].fillna(130))
        x = df['calories'].mean()
        df['calories'].fillna(x)