# --------------Without external libraries--------------

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	res = []
	for child in a :
		if len(child) != len(b) :
			return -1 
		else :
			cnt = 0 
			for i in range(len(child)) :
				cnt += (child[i] * b[i])
			res.append(cnt)
	return res 

# --------------With numpy--------------
import numpy as np

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	try :
		return np.dot(a,b)
	except Exception as e :
		return -1