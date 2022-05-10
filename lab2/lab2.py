import string

# Splits a string by character


def split(s, character):
	res = [""]
	for c in s:
		if c == character:
			res.append("")
		else:
			res[-1] += c
	return res


print(split(",Hello, World!,", ","))


def check_password_strength(password):
	if len(password) < 10:
		return False

	if not any(c.isupper() for c in password):
		return False

	if not any(c.islower() for c in password):
		return False

	if not "!" in password:
		return False

	return True


print(check_password_strength("HelloWorld!"))
print(check_password_strength("HelloWorld"))
