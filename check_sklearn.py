import sys
try:
	import sklearn
	print("sklearn", getattr(sklearn, "__version__", "unknown"))
except Exception as e:
	print("sklearn import error", e)
print(sys.executable)
