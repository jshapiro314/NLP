import re
#I ASSUME IPV4#
pattern = re.compile('\\b(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\b')

userInput = input('Please enter the IP address for testing\n')

address = pattern.match(userInput)

if address:
    print("Valid IP Address")
else:
    print("Invalid IP Address")