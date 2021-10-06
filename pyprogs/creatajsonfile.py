#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:43:48 2021

@author: elham
"""
list = [(1,4),(3,5),(0,6),(5,7),(3,9),(5,9),(6,10),(8,11),(8,12),(12,16)]

with open('Set_s_f_time.txt', 'w') as fp:
    fp.write('\n'.join('%s %s' % x for x in list))