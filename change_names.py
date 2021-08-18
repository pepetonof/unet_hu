# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 23:23:59 2021

@author: josef

Cambio de nombre de los archivos de un folder en específico
"""

import os

def change_names(cont:int, folder:str):
    os.chdir(os.getcwd() + '\\' + folder)
    filenames=sorted(os.listdir())
    
    for name in filenames:
        os.rename(name, folder+str(cont)+'.png')
        cont+=1
    
    os.chdir('..')
    print('Archivos renombrados')
