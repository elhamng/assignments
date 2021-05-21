#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:42:16 2021

@author: elham
"""
from examples_python import DicExamples, TupleExamples, ListExamples, MapExamples, NumPyExamples
from examples_python import PanDasExamples

def main():
    dict_examples = DicExamples()
    #dict_examples.example_11()
    #dict_examples.example_14()

    
    tuple_example = TupleExamples()
    #tuple_example.example_01()
    
    list_exmaples = ListExamples()
    #list_exmaples.example_01()
    #map_examples = MapExamples()
    #map_examples.example_06()
    my_examples = NumPyExamples()
    #my_examples.example_01()
    pandas_examples = PanDasExamples()
    pandas_examples.example_01()
    
main()