import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
import time

# Train SVM
def train_svm(X_train,y_train, X_test, y_test, kernel='poly'):
	clf = svm.SVC(class_weight='balanced', kernel=kernel)
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	cm = confusion_matrix(y_test, y_pred)
	return cm, clf
	
def analyze_CM(cm):
	tn, fp, fn, tp = cm.astype(float).ravel()
	acc = (tp+tn)/(tn+fp+fn+tp)
	sen = tp/(tp+fn)
	spec = tn/(tn+fp)
	prec = tp/(tp+fp)
	F1 = 2 * (prec * sen) / (prec + sen)
	print(cm)
	print('accuracy: ',acc)
	print('sensitivity:', sen)
	print('specificity:', spec)
	print('precision:', prec)
	print('F1:', F1)

# X and y must not be shuffled
# Calculates Latency, Repeated FP, Repeated FN
def calc_other(y, X, clf, dt):
	y_pred = clf.predict(X)

	latency = []
	idx = np.where(y==1)[0]
	if len(idx) > 0:
		idx_break = np.where(-idx[:-1]+idx[1:]>1)[0]+1
		idx_break = np.insert(idx_break,0,0)
		start = 0
		for idx_seizure in idx_break:
			start = idx[idx_seizure]
			lat = 0
			while y[start]==1:
				if y_pred[start] != 1:
					lat += 1
				else:
					break
				start += 1
			if y[start] == 0: # Never detected the seizure
				lat = np.inf
			latency.append(lat)
		latency = np.array(latency)*dt
	
	y_d = y - y_pred # 1 = FN, -1 = FP
	# Calculate repeat FN
	FN_list = []
	idx = np.where(y_d == 1)[0]
	if len(idx) > 1:
		count = 1
		for i in range(len(idx[:-1])):
			if idx[i]+1 == idx[i+1]:
				count += 1
			else:
				FN_list.append(count)
				count = 1
			if i + 1 == len(idx[:-1]):
				FN_list.append(count)
	elif len(idx) == 1:
		FN_list.append(1)

	FP_list = []
	idx = np.where(y_d == -1)[0]
	if len(idx) > 1:
		count = 1
		for i in range(len(idx[:-1])):
			if idx[i]+1 == idx[i+1]:
				count += 1
			else:
				FP_list.append(count)
				count = 1
			if i + 1 == len(idx[:-1]):
				FP_list.append(count)
	elif len(idx) == 1:
		FP_list.append(1)
	return latency, FN_list, FP_list

def analyze_latency(latency):
	print(latency)
	if len(latency)>0:
		print('Avg latency: ', np.mean(latency))
		print('Min latency: ', np.amin(latency))
		print('Max latency: ', np.amax(latency))
		print('Std latency: ', np.std(latency))

def SMV_cross_validate(X_2d, y, dt, k_fold, random_state = 0, kernel = 'linear'):
	print('Testing kernel: ', kernel)
	kernel_start = time.time()
	X_2ds, ys = shuffle(X_2d, y, random_state=random_state)
	kf = KFold(n_splits=k_fold, random_state=random_state)
	kf.get_n_splits(X_2ds)
	for idx, (train_index, test_index) in enumerate(kf.split(X_2ds)):
		print("Iteration: ",idx)
		k_start = time.time()
		X_train, X_test = X_2ds[train_index], X_2ds[test_index]
		y_train, y_test = ys[train_index], ys[test_index]
		if idx == 0:
			cm, clf = train_svm(X_train,y_train, X_test, y_test, kernel=kernel)
			latency, FN_list, FP_list = calc_other(y, X_2d, clf, dt)
			if kernel == 'linear':
				weight = np.squeeze(clf.coef_)
			else:
				weight = []
		else:
			cm_temp, clf = train_svm(X_train,y_train, X_test, y_test, kernel=kernel)
			cm += cm_temp
			latency_temp, FN_list_temp, FP_list_temp = calc_other(y, X_2d, clf, dt)
			latency = np.hstack([latency, latency_temp])
			FN_list = np.hstack([FN_list, FN_list_temp])
			FP_list = np.hstack([FP_list, FP_list_temp])
			if kernel == 'linear':
				weight = np.vstack([weight, np.squeeze(clf.coef_)])
		print('Time: ', time.time()-k_start)
		print(cm)
		print(latency)
		print(FN_list)
		print(FP_list)
   	print('---------------------------------------------------------')
	print('Result for kernel: ', kernel)
	print('Avg Time: ', (time.time()-kernel_start)/k_fold)
	analyze_CM(cm)
	analyze_latency(latency)
	return weight