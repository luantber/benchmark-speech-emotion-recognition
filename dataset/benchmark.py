from datetime import datetime
import csv

class Benchmark:
    def __init__(self,name,model_name,result):
        self.name = name
        self.model_name = model_name
        self.result = result #dictionary
    
    def write(self,postfix="_benchmark.csv"):
        print(self.name, postfix)
        file_name = self.name + postfix
        with open(file_name,"a") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["Name"] + list(self.result.keys()) + ["Timestamp"] )    
            writer.writerow([self.model_name ] + [ "{:.4f}".format(res) for res  in list(self.result.values())] + [datetime.now()] )

    



