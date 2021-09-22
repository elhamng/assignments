#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:18:26 2021

@author: elham
"""

# Read the name and mark of at least 3 students
#Rank the top 3 students with higest marks
# Give each student cash rewards $500 for first $300 for second and $ 100 for third 
#Appreciate student who secured 950 marks and abov 

import operator

def readStudentDetails():
    print('Enter the number of students:')
    numberOfStudents = int(input())
    studentRecord ={}
    for i in range(0,numberOfStudents):
        print('Enter the name of student') 
        name = input()
        print('Enter the mark of student')
        marks = int(input())
        studentRecord[name] =  marks
        print()
    return studentRecord
     
    
def rankStudents(studentRecord):
    sortedStudentRecord = sorted(studentRecord.items(), key = operator.itemgetter(1), reverse= True)
    print(sortedStudentRecord)
    print("{} has secured first rank, scoring {} marks".format(sortedStudentRecord[0][0],sortedStudentRecord[0][1]))
    print("{} has secured second rank, scoring {} marks".format(sortedStudentRecord[1][0],sortedStudentRecord[1][1]))
    print("{} has secured third rank, scoring {} marks".format(sortedStudentRecord[2][0],sortedStudentRecord[2][1]))
    print()
    return(sortedStudentRecord)   
    
    
    
    
def rewardStudents(sortedStudentRecord,reward):
    print('{} has recived a cash reward of ${}'.format(sortedStudentRecord[0][0],reward[0]))
    print('{} has recived a cash reward of ${}'.format(sortedStudentRecord[1][0],reward[1]))
    print('{} has recived a cash reward of ${}'.format(sortedStudentRecord[2][0],reward[2]))
    print()

def appreciateStudents(sortedStudentRecord):
    for record in sortedStudentRecord:
        if record[1] >= 950:
            print('Congratulation on scoring {} marks,{}'.format(record[1],record[0]))
        else:
            break 
    print()


studentRecord = readStudentDetails()
sortedStudentRecord = rankStudents(studentRecord)
reward = (300,200,100)
rewardStudents(sortedStudentRecord,reward)
appreciateStudents(sortedStudentRecord)

