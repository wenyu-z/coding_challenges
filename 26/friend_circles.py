#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 00:09:52 2017

@author: Wenyu
"""

#%%

"""
Given an array that represnets friendship status between students, if (i,j) is
"Y" then they are friends and such relationsihp is transisitive, meaning if i&j
are friends, j&k are friends, then i&k are friends, although not necessarily 
explicitly writen in the data. A student is at minimum his/her own friend. 

Write a function to find out all circles in the data
"""

def  friendCircles(friends):
    no_person = len(friends)
    circles = []
    for i in range(no_person):
        j = i
        
        while j < no_person:
        
            if friends[i][j] == 'Y':
                if any([i in cir for cir in circles]) & any([j in cir for cir in circles]):
                    cir_indi = [it for (it, val) in enumerate(circles) if i in val][0]
                    cir_indj = [it for (it, val) in enumerate(circles) if j in val][0]
                
                    ciri = circles[cir_indi]
                    cirj = circles[cir_indj]
                    
                    newcir = list(set(ciri).union(cirj))
                    
                    circles.remove(ciri)
                    if cirj in circles:
                        circles.remove(cirj)
                    
                    circles.append(newcir)
                    
                elif any([i in cir for cir in circles]):
                    cir_ind = [it for (it, val) in enumerate(circles) if i in val][0]
                    circles[cir_ind].append(j)
                    
                elif any([j in cir for cir in circles]):
                    cir_ind = [it for (it, val) in enumerate(circles) if j in val][0]
                    circles[cir_ind].append(i)
                    
                else:
                    newcir = [i,j]
                    circles.append(newcir)
            else:
                pass
            
            j += 1
            
    circles = [list(set(cir)) for cir in circles]
    return len(circles)

friends = ['YYNN', 'YYYN', 'NYYN', 'NNNY']
print friendCircles(friends)

# returns 2

#%%
