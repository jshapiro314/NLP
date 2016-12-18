
file = open('en-universal-devMod.txt')
for line in file:
	array = line.split("\t")
	if len(array) == 1:
		print()
	else:
		print(array[0])