
import numpy as np
#https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1

def precision_tensor(matrix):
    #what proportion of predicted Positives is truly Positive? 
    # spam cuantos fueron clasificados falsamente ( perdidad de informacion)
    # print( "diag" , np.diag(matrix) )
    # print( "sums" ,  matrix.sum(axis=0).reshape(-1) )
    res = np.diag(matrix) / matrix.sum(axis=0).reshape(-1)
    res = np.nan_to_num(res)
    return res 


def recall_tensor(matrix):
    #what proportion of actual Positives is correctly classified? 
    #cancer ( cuantos pacientes se les dijo negativo pero si tenían)
    # print( "diag" , np.diag(matrix) )
    # print( "sums" ,  matrix.sum(axis=1).reshape(-1) )
    res = np.diag(matrix) / matrix.sum(axis=1).reshape(-1)
    res = np.nan_to_num(res)
    return res


def f1_tensor(matrix):
    p=precision_tensor(matrix) 
    r=recall_tensor(matrix)

    # print ( p )
    # print ( r )
    res =  ( 2 * ( p * r ) / ( p + r) )
    res = np.nan_to_num(res)

    return  res


def metric(matrix, operation ,modo): ## rename 
    tensor = operation(matrix)
    count = matrix.sum(axis=0)
    if modo=="macro":
        return tensor.mean()
    elif modo=="weight":
        return (tensor * count ).sum()/ count.sum()
    elif modo=="micro":
        return np.diag(matrix).sum() / matrix.sum()


def f1(matrix,modo="macro"):
    return metric( matrix, f1_tensor , modo )
    
def precision(matrix,modo="macro"):
    return metric( matrix, precision_tensor , modo )

def recall(matrix,modo="macro"):
    return metric( matrix, recall_tensor , modo )

def accuracy(matrix):
    return f1(matrix,modo="micro")
