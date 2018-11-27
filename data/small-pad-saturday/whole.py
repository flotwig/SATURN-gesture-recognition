import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import sys

try:
	fn = sys.argv[1] + ".csv"
	f = open(fn)
	csv_f = csv.reader(f)

	data = []
	for row in csv_f:
		data.append(float(row[0]))

	plt.plot(data)
	plt.title(fn)
	plt.show()
except:
	print "error. check filename"