#https://www.geeksforgeeks.org/call-a-function-by-a-string-name-python/

# Importing a module random
import random

# Using random function of random
# module as func
func = "random.random"

# calling the function and storing
# the value in res
exec(f"x = {func}")
res = x()

# Printing the result
print(res)