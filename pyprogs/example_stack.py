#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 07:37:44 2021

@author: elham
"""

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

exp = '{[()]}'
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
            print(i)
            index = clBracket.index(i)
            print(index)
            if opBracket[index] == stack[len(stack)-1]:
                x = myStack.pop()
                if x is None:
                    valid = False
                print(x)
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
    

print(check(exp))