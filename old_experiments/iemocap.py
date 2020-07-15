import get_files
import prepro
import cv2
import matplotlib.pyplot as plt
import threading 

x, y = get_files.get_dataset()
x_s = [None] * len(x)

def function_g(x,n_hilo,parte ):
    print("start")
    i = 0
    for xi in x:
        # x_s.append( prepro.get_spec(xi) ) 
        x_s[ n_hilo*parte + i ] =  prepro.get_spec(xi,size=128)
        i+=1
    print("fin")

def parallel_generation(x,n_hilos = 4):
    size = len(x)
    parte = int(size/n_hilos)

    hilos = []
    for h in range(n_hilos):
        if h==(n_hilos-1):
            #print ( h * parte , size )
            th = threading.Thread(target=function_g, args=(x[h*parte: size ], h , parte ))
        else:
            #print ( h * parte , (h+1)*parte )
            th = threading.Thread(target=function_g, args=(x[h*parte: (h+1)*parte ], h , parte ))
        th.start()
        hilos.append(th)


    for h in hilos:
        h.join()



parallel_generation(x)
print ( len(x) , len(y))
print( len(x_s) )

import pickle

pickle.dump( ( x_s, y ) , open( "iemocap128.pk", "wb" ) )