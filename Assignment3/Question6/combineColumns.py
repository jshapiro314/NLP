
file = open('en-universal-new-train.txt')
for line in file:
	array = line.split("\t")
	if len(array) == 1:
		print()
	else:
		print(array[1],"	",array[3],"-",array[4],sep="")

