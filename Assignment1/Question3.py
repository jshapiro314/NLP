userInput = int(input('Please enter a number\n'))
result = 0

while userInput > 0:
	result += (userInput % 10)
	userInput -= (userInput % 10)
	userInput /= 10

print ("Sum of digits = ",result)