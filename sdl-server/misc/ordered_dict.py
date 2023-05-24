# A Python program to demonstrate working of OrderedDict
from collections import OrderedDict
from queue import PriorityQueue



print("\nThis is an Ordered Dict:\n")
od = OrderedDict()
od['a'] = 1
od['b'] = 2
od['c'] = 3
od['d'] = 4

for indx, key,val in od.items():
	print(key)
	



for key, value in od.items():
	print(key, value)
