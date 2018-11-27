import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import sys

if len(sys.argv) != 3:
	print "wrong parms"
	sys.exit()

gesture = sys.argv[1]
start = int(sys.argv[2])
interest = []

try:
	fn = gesture + ".csv"
	f = open(fn)
	csv_f = csv.reader(f)

	data = []
	for row in csv_f:
		data.append(float(row[0]))

	interest = data[start : (start + 500)]
	plt.plot(interest)
	plt.title(fn)
	plt.show()
	
	arr = open('window_array.txt', 'w')
	arr.write(str(interest))
	arr.close()
	
except Exception, err:
	print "error. check filename"
	print err