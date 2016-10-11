import re
#parse text and place into list
pattern = "[\\w]+"
text = input("Enter your list of numbers:   ")

numbers = (re.findall(pattern, text))
numbers = [int(i) for i in numbers]
#sort list
#print(numbers)
numbers.sort()
#print(numbers)

#iterate over list, calculating total and printing each item
itemCount = 0
tokenCount = len(numbers)
typeCount = 0
i=0
while i < tokenCount:
	itemCount = numbers.count(numbers[i])
	print(numbers[i], " ", itemCount)
	i+=itemCount
	typeCount+=1
print("Tokens: ", tokenCount)
print("Types: ", typeCount)
