import re
#read in file
file = open("uncorpora_plain_20090831.tmx", "r")
count = 0

for line in file:
	if re.search("[Hh]uman [Rr]ights", line):
		count+=1

print(count, " lines contain Human Rights")