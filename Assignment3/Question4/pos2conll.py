#from itertools import izip

with open("en-universal-devModTagged5.txt") as file, open("en-universal-dev.txt") as file2: 
    for x, y in zip(file, file2):
        array1 = x.split("\t")
        array2 = y.split("\t")
        if len(array1) == 1:
        	print()
        else:
        	array3 = array1[1].split("-")
        	print(array2[0],array2[1],array2[2],array3[0],array3[1],array2[5],array2[6],array2[7],array2[8],array2[9],sep="	",end="")
        