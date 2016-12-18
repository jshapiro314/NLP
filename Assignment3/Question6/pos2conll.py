#from itertools import izip

with open("en-universal-test-TAGGED.txt") as file: 
    for x in file:
        array1 = x.split("\t")
        if len(array1) == 1:
        	print()
        else:
        	array3 = array1[2].split("-")
        	print(array1[0],array1[1],"_",array3[0],array3[1],"_","_","_","_","_",sep="	")
        