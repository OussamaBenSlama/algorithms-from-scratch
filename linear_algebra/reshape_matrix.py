# --------------Without external libraries--------------
def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	
	n = len(a)
	m = len(a[0]) 

	p,q = new_shape

	if n*m != p*q :
		return []

	flat = [e for row in a for e in row]
	reshaped_matrix = []

	for i in range(p) :
		row = flat[i*q : (i+1)*q]
		reshaped_matrix.append(row)
	return reshaped_matrix


# --------------With numpy--------------
import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	
	n = len(a)
	m = len(a[0]) 

	if n*m != new_shape[0]*new_shape[1] :
		return []
	else :
		a = np.array(a)
		reshaped_matrix = np.reshape(a, new_shape).tolist()
	return reshaped_matrix