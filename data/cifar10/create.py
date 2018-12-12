import os
import sys

f = open("database.txt")
lines = f.readlines()
f.close()

tr = open("fulltrain.txt","w")
te = open("fulltest.txt","w")

for ln in lines:
	if ln.startswith("train"):
		tr.write("%s" % ln)
	elif ln.startswith("test"):
		te.write("%s" % ln)
	else:
		print("ERROR: " + ln)

tr.close()
te.close()
