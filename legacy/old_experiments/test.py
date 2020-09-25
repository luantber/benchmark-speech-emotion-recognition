import get_files
import prepro 
import pandas as pd 
import matplotlib.pyplot as plt 
import pickle
x,y = get_files.get_dataset()

duraciones = []
for i in x:
    print ( i )

    duraciones.append(prepro.get_spec(i))
    break


# print ( len(duraciones)) 
# pickle.dump( duraciones , open( "duraciones.pk", "wb" ) )

# apo  = pd.Series( duraciones )
# apo.plot.hist(bins=25)
# plt.title('Commute Times for 1,000 Commuters')
# plt.xlabel('Counts')
# plt.ylabel('Commute Time')
# plt.show()