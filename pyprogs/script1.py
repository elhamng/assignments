#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 11:05:29 2021

@author: elham
"""
## Build and connect to database

#import psycopg2
import psycopg2 

def creat_table():
    conn = psycopg2.connect("dbname='database1' user ='postgres' password = 'EnjoyMac' host = 'localhost' port ='5432' ")
    cur = conn.cursor()
    cur.execute("CREATE TABLE  IF NOT EXISTS store(item  TEXT, quantity INTEGER, price REAL)")
    cur.execute("INSERT INTO store VALUES('Wine Glass')")
    conn.commit()
    conn.close()



def insert(item,quantity,price):
    conn = psycopg2.connect("dbname='database1' user ='postgres' password = 'EnjoyMac' host = 'localhost' port ='5432' ")
    cur = conn.cursor()
    cur.execute("INSERT INTO store VALUES(%s,%s,%s)", (item,quantity,price))
    conn.commit()
    conn.close()
 


def view():
    conn = psycopg2.connect("dbname='database1' user ='postgres' password = 'EnjoyMac' host = 'localhost' port ='5432' ")
    cur = conn.cursor()
    cur.execute("SELECT * FROM store")
    rows = cur.fetchall()
    conn.close()
    return rows
   

def delete(item):
    conn = psycopg2.connect("dbname='database1' user ='postgres' password = 'EnjoyMac' host = 'localhost' port ='5432' ")
    cur = conn.cursor()
    cur.execute("DELETE FROM store WHERE item =%s", (item,))
    conn.commit()
    conn.close()
    
def update(quantity,price,item):
    conn = psycopg2.connect("dbname='database1' user ='postgres' password = 'EnjoyMac' host = 'localhost' port ='5432' ")
    cur = conn.cursor()
    cur.execute("Update store SET quantity=%s,price=%s WHERE item = %s", (quantity,price,item))
    conn.commit()
    conn.close()


creat_table()
insert('Apple',10,2)
