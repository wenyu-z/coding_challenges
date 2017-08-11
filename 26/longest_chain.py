#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 00:03:45 2017

@author: Wenyu
"""

#%%

"""
 Given a list of words/strings, for each word you can remove one letter
 only when it results in another word that exists in the list, which forms
 a chain of length 2. Continue to remove one word at a time, resulting in 
 words in the list, and the chain increase every time a new word is found

 Write a function that can find the longest chain possible in the list of 
 words
"""

"""
The code currently does not have dynamic programming, and the efficiency is
not optimal.

"""

def removalsuccess(compareword, word):
    for letter in compareword:
        if letter not in word:
            return False
            break
    return True

def  longestChain(words):
    words.sort(key=lambda word: len(word))
    
    maxchain = [1]*len(words)
    
    for i in range(1, len(words)):
        word = words[i]
        for j in range(i):
            compareword = words[j]
            if len(word) - len(compareword) == 1:
                if removalsuccess(compareword, word):
                    maxchain[i] = max(maxchain[i], maxchain[j]+1)
        
    return max(maxchain)
    
    
words = ['a', 'b', 'ab', 'bca', 'bda', 'bdca']
print longestChain(words)

# returns 4




#%%