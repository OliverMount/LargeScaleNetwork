import numpy as np
from collections import Counter
from sklearn.datasets import load_iris, load_breast_cancer 
from sklearn.model_selection import train_test_split

def dist(a, b, dtype = "euclidean"):
	""" 
	이 함수는 두 개 레코드 사이의 거리를 반환한다. 

	Examples
	--------
	>>> dist(np.array([1,2,3,4]), np.array([2,3,1,1]), "euclidean")
	3.0
	>>> dist(np.array([1,2,3,4]), np.array([2,3,1,1]), "manhattan")
	3.872983346207417
	>>> dist(np.array([1,2,3,4]), np.array([2,3,1,1]), "cosine")
	0.29289321881345254
	"""
	import numpy as np 
	import math 
	
	# Minkowskii Distance
	if dtype.lower() == "euclidean":
		m = 2
		d = pow(sum(pow(a-b, m)), 1/m)

	if dtype.lower() == "manhattan":
		m = 1
		d = sum(abs(pow(a-b, m)))

	# Cosine Distance (Cosine Similarity)
	if dtype.lower() == "cosine":	
		d = 1 - (np.dot(a, b) / math.sqrt((np.dot(a, a) * np.dot(b, b))))
		# d는 0~1 사이의 값을 얻으며, 0은 벡터가 서로 100% 유사함을 의미한다.

	return d


# * ----- * ----- * ----- * ----- * ----- * ----- 
# 2. 데이터 불러오기

X, y = load_iris(return_X_y = True)
#X, y = load_breast_cancer(return_X_y = True)


# * ----- * ----- * ----- * ----- * ----- * ----- 
# 3. 훈련-테스트 세트 구성하기

tsize = 0.2
seed = 5
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=tsize, random_state=seed)


# * ----- * ----- * ----- * ----- * ----- * ----- 
# 4. kNN 적용하기
# 4.1. 주요 파라미터 설정

K = 2
rows = Xtest.shape[0]
cols = Xtrain.shape[0]

# 4.2. 거리 구하기 : 두 레코드 사이의 거리 구하기

d = np.zeros((rows, cols))
for i in range(rows):
	for j in range(cols):
		d[i, j] = dist(Xtest[i], Xtrain[j])

# 4.3. 가까운 레코드 찾기 : 테스트 레코드와 가장 가까운 k개의 훈련 레코드 인덱스 찾기

idx = np.zeros((rows, K))
for i in range(rows):
	idx[i] = np.argsort(d[i])[:K]

# 4.4. 타겟 데이터 적용하기
idx = idx.astype(int)	# for indices type error prevention

return_y = np.zeros((rows, K))
for i in range(rows):
	return_y[i] = ytrain[idx[i]]
print("return_y[0:5] :\n", return_y[0:5])

# 4.5. 메이저 타겟 선정하기 : 선정 된 k개의 타겟 중 가장 우세한 타겟 레이블을 선정하여 테스트 레코드에 적용

return_maj = np.zeros((rows, 1))
for i in range(rows):
	c = Counter(return_y[i])
	value, count = c.most_common()[0]
	return_maj[i] = value
ypred = return_maj.astype(int)
print("ypred[0:5] :\n", ypred[0:5])

# * ----- * ----- * ----- * ----- * ----- * ----- 
# 5. 모델 성능 평가 
# 5.1. Confusion matrix

def myConfusionMatrix(ytest, ypred, margins=False):
	# Confusion Matrix
	ylabel = np.unique(ytest)  
	K = len(ylabel)		# K: Number of classes
	CM = np.zeros((K, K))
	for i in range(len(ytest)):
		CM[ytest[i]][ypred[i]] += 1
	CM = CM.astype(int)

	if margins:	# Add row/column margins (subtotals)
		# Subtotal_bottom
		s = 0
		lst = []
		for i in range(len(ylabel)):
			for j in range(len(ylabel)):
				s = s + CM[j][i]
			lst.append(s)
			s = 0
		# Print
		print("Confusion Matrix\n----------------")
		print("Predicted {}\tAll".format(ylabel))	# line 1
		print("Actual") 				# line 2
		for i in range(len(ylabel)):			# line 3~
			print("{}\t  {}\t{}".format(ylabel[i], CM[i], sum(CM[i])))
		print("All\t  {}\t{}".format(lst, sum(lst)))  # line ~fin.
	
	return CM if not margins else ""

print(ytest)
print(ypred)

print(ytest.shape)
print(ypred.shape)

print(myConfusionMatrix(ytest, ypred, margins=True))

# 5.2. Accuracy

def myAccu(ytest, ypred):
	return np.mean(ytest == ypred.T)

print("Accuracy score is : ", myAccu(ytest, ypred))
