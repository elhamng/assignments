#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 11:44:27 2021

@author: elham
"""

x = 10

#
    
# example 01    
class Person:
    def __init__(self,name, address):
        self.name = name
        self.address = address
    
    def whoami(self):
        return "you are " + self.name
    def wheredoIlive(self):
        return self.address


#hiding  data field using two leading underscores. self.__name         
    

# example 02
class BankAccount:
    def __init__(self, name, money):
        self.__name = name
        self.__balance = money
    
    def deposit(self,money):
        self.__balance += money
    
    def withdraw(self,money):
        if self.__balance > money:
            self.__balance -= money
            return money
        else:
            return "there is not sufficient funds"
    
    def checkbalance(self):
        return self.__balance
    


# example 03            
#convert int number to roman 
class NumToRomanConvert:
    def __init__(self):
        self.num_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
           (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
    
    def numtoroma(self , num):        
        rom_num = ''
        while num > 0:
            for k,v in self.num_map:
                while num >= k:
                    rom_num += v
                    num -=k
        return(rom_num)
    

 #example 04
#convert roman to integer
class RomtoNumConverter:
    def __init__(self):
        self.num_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    def romtonum(self,rom):
        num = 0
        x = str.upper(rom)
        #for i in range(len(x)):
            # if i > 0 and self.num_map[x[i]] > self.num_map[x[i - 1]]:
            #     num += self.num_map[x[i]] - 2 * self.num_map[x[i - 1]]
            # else:
                
            #     num += self.num_map[x[i]]
        while x:
            print(x)
            if len(x) == 1 or self.num_map[x[0]] >= self.num_map[x[1]]:
                num += self.num_map[x[0]]
                x = x[1:]
            else:
                num += self.num_map[x[1]] - self.num_map[x[0]]
                x = x[2:]    
                    
        return num   
#example 05
# Python3 code to Check for  
# balanced parentheses in an expression 
class CheckParanteses:
    def __init__(self):
        self.map = {"(":")","[":"]","{":"}"}
    def is_valid_parenthese(self,string):
        stack = []
        for parenthese in string:
             if parenthese in self.map:
                stack.append(parenthese)
                print(stack)
             elif parenthese not in self.map:
                print(parenthese)
          
             elif len(stack)== 0 or self.map[stack.pop()]!=parenthese:
                print(stack)
                return False
              
                
        return len(stack)==0

        #return balanced     
class Stack():
    def __init__(self):
        self.items = []
        
    def push(self,item):
        self.items.append(item)
        return self.items
    
    def pop(self):
        if self.isEmpty():
            return None
        else:
            return self.items.pop()

    def isEmpty(self):
        return len(self.items) == 0
    
    def printStack(self):
        print(self.items)
    
myStack = Stack()


def check (exp):
    valid = True
    opBracket = ['(','[','{']
    clBracket = [')',']','}']
    
    for i in str(exp):
        if i in opBracket:
            stack = myStack.push(i)
            #last_push = i
            print(stack)
        elif  i in clBracket:
            index = clBracket.index(i)
            if opBracket[index] == stack[len(stack)-1]:
                x = myStack.pop()
                if x is None:
                    valid = False
            else:
                valid = False
        if valid == False:
            break

    myStack.printStack()

    if myStack.isEmpty():
        valid = True
    else:
        valid = False
    print(valid)
    result = 'Valid Expresion' if valid else 'Invalid Expression'
    return result  
    

             

def main():
    # how to use example 01
    # p1=Person('Tom','Boerhaavelaan') 
    # p2=Person('Elham','Jan Steenstraat')
    # print(p1.name)
    # print(p2.whoami())
    # print(p1.whoami())
    # print(p1.wheredoIlive())
    # p1.address = 'DelftJan'
    
    # print(p1.wheredoIlive())
    
    # print(p2.wheredoIlive())

    # # how to use example 02    
    # b2 = BankAccount('tim', 400)
    # print(b2.withdraw(500))
    # b2.deposit(500)
    # print(b2.checkbalance())
    # print(b2.withdraw(800))
    # print(b2.checkbalance()) 
    # #print(b2.name)   This a private field

    # # useexample 03
    # ntor = NumToRomanConvert()
    # print(ntor.numtoroma(235))
    
    # useexample 04
    #rton = RomtoNumConverter()
   # print(rton.romtonum('MMMCMLXXXVI'))
    #print(rton.romtonum('MMMM'))
    #print(rton.romtonum('clmc'))
    #useexample05
    # checkbalance = CheckParanteses()
    # print(checkbalance.is_valid_parenthese("{[]{()}}"))
    
    #
    exp = '{5-[3+(2+4)*5)-8'
    print(check(exp)) 
    
main()