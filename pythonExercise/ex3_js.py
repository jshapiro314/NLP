import re

#create a person class
class Person:
	def __init__(self, fName, lName, phone, email, homePages, age, employer):
		self.fName = fName
		self.lName = lName
		self.phone = phone
		self.email = email
		self.homePages = homePages
		self.age = age
		self.employer = employer

	def __str__(self):
		return "{First Name: %s | Last Name: %s | Phone: %s | Email: %s | Homepage: %s | Age: %s | Employer: %s}" % (self.fName, self.lName, self.phone, self.email, self.homePages, self.age, self.employer)

	def __repr__(self):
		return "{First Name: %s | Last Name: %s | Phone: %s | Email: %s | Homepage: %s | Age: %s | Employer: %s}" % (self.fName, self.lName, self.phone, self.email, self.homePages, self.age, self.employer)

#Fetch name, phone, email, homepages, age, and employer of each person & put in a dictionary

#For phone: ([\(+]*[0-9]{3,4}[\)\-,]*){3}
#For email: [\w\d]+(@|\sat\s)[a-z]+(\.|\sdot\s)[a-z]+
#For age: ((I'm)|(I am)|and)(\s)*([0-9]+)    --- then group 5
#For website: \s((http:\/\/)?(www\.)?[a-z0-9]+\.[a-z]{3})     --- group 1
#For employer: \b((a|for|at)\s([A-Z]?[a-z]+(\s[A-Z]*[a-z]+)?|[A-Z]+))   --- first match, group 3
#For name: (I am\s|I'm\s)([A-Z][a-z]+)\s([A-Z][a-z]+)? --- 2nd and 3rd group

nPat = '''(I am\s|I'm\s)([A-Z][a-z]+)\s?([A-Z][a-z]+)?'''
pPat = '''([\(+]*[0-9]{3,4}[\)\-,]*){3}'''
ePat = '''[\w\d]+(@|\sat\s)[a-z]+(\.|\sdot\s)[a-z]+'''
wPat = '''\s((http:\/\/)?(www\.)?[a-z0-9]+\.[a-z]{3})'''
aPat = '''((I'm)|(I am)|and)(\s)*([0-9]+)'''
bPat = '''\\b((a|for|at)\s([A-Z]?[a-z]+(\s[A-Z]*[a-z]+)?|[A-Z]+))'''

#ASSUMING THAT EACH PERSON'S BIO IS ON A SEPARATE LINE

#First read in text file
bios = []
peopleFile = open("regex.txt", "r")
for line in peopleFile:
	bios.append(line)
#print(bios)

#start parsing bios
people = {}
for person in bios:
	name = re.search(nPat, person)
	fName = name.group(2)
	lName = name.group(3)
	phone = re.search(pPat, person).group(0) if re.search(pPat, person) else None
	email = re.search(ePat, person).group(0) if re.search(ePat, person) else None
	homePages = re.search(wPat, person).group(1) if re.search(wPat, person) else None
	age = re.search(aPat, person).group(5) if re.search(aPat, person) else None
	employer = re.search(bPat, person).group(3) if re.search(bPat, person) else None
	newPerson = Person(fName, lName, phone, email, homePages, age, employer)
	people[fName] = newPerson

print(people)


