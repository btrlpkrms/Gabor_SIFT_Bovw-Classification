import numpy as np
import cv2
import glob
import math
import operator
from sklearn.cluster import KMeans
trains = []
queries = []

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors


def getFolders(string):
    folders = glob.glob(string)
    imagenames_list = []
    for folder in folders:
        for f in glob.glob(folder+'/*.jpg'):
            imagenames_list.append(f)
    return imagenames_list

imageNames = getFolders("dataset/train/*")
queryNames = []

for f in glob.glob("dataset/query/*"):
    queryNames.append(f.split("\\")[1].split("_")[0])

def getTrains(imagenames):
    trains = []
    for image in imagenames:
        trains.append(cv2.imread(image))
    return trains
trains = getTrains(imageNames)

def getQuery(string):
    trains = [cv2.imread(file) for file in glob.glob(string+"*.jpg")]
    return trains
queries = getQuery("dataset/query/")

def normalizeVector(list):
    vector = []
    for x in list:
        normalized = (x-min(list))/(max(list)-min(list))
        vector.append(normalized)
    return vector

def build_filters():
    filters = []
    ksize = 21 
    for theta in np.arange(0, np.pi, np.pi /8):
        for sigm in np.arange(2,12,2):
            params = {'ksize': (ksize, ksize), 'sigma': sigm, 'theta': theta, 'lambd': 10.0,
                      'gamma': 0.5, 'psi': 0, 'ktype': cv2.CV_32F}
            kern = cv2.getGaborKernel(**params)
            filters.append((kern, params))
            h, w = kern.shape[:2]
            kern = cv2.resize(kern, (3 * w, 3 * h), interpolation=cv2.INTER_CUBIC) 
    return filters

def process(img, filters):
    vector = []
    for kern, params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        a = np.mean(fimg)
        vector.append(a)
    return vector

filters = build_filters()

def apply_filter_all(list,imageNames):
    filtered = []
    a = []
    for i in imageNames:
        a.append(i.split("\\")[2].split(".")[0])
    for i in range(len(list)):
        x = process(list[i], filters)
        x.append(a[i])
        filtered.append(x)
        x =[]
    return filtered

def apply_filter_query(list):
    filtered = []
    for i in range(len(list)):
        x = process(list[i], filters)
        x.append(queryNames[i])
        filtered.append(x)
    return filtered

def apply_SIFT_train(list):
    features = []
    b = []
    for i in imageNames:
        b.append(i.split("\\")[2].split("_")[0])       
    for i in range(len(list)):
        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(list[i],None)
        a = np.mean(des,axis=0).tolist()
        a.append(b[i])
        features.append(a)
    return features

def apply_SIFT_query(list):
    features = []     
    for i in range(len(list)):
        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(list[i],None)
        a = np.mean(des,axis=0).tolist()
        a.append(queryNames[i])
        features.append(a)
    return features

def get_Class_Based_Accuracy(featureTrain,featureQuery):
    predictions=[]
    k = 5
    for x in range(len(featureQuery)):
    	neighbors = getNeighbors(featureTrain, featureQuery[x], k)
    	result = getResponse(neighbors)
    	predictions.append(result)
    	print('> predicted=' + repr(result) + ', actual=' + repr(featureQuery[x][-1]))
    accuracy = getAccuracy(featureQuery, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    

def get_All_Descriptors(list):
    des_list = []
    for i in range(len(list)):
        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(list[i],None)
        des_list.append((imageNames[i].split("\\")[2].split("_")[0], des))  
    return des_list

def query_Descriptors(list):
    query = []
    for i in range(len(list)):
        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(list[i],None)
        query.append((queryNames[i],des))
    return query

def stackDescriptors(des_list):
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor)) 
    return descriptors
def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram
print("Gabor Results")    
trainsResults = apply_filter_all(trains,imageNames)
queriesResults = apply_filter_query(queries)
get_Class_Based_Accuracy(trainsResults,queriesResults)
print("Average SIFT")
siftFeaturesTrain = apply_SIFT_train(trains)
siftFeaturesQuery = apply_SIFT_query(queries)
get_Class_Based_Accuracy(siftFeaturesTrain,siftFeaturesQuery)

print("Bag Of Visual Words Accuracy")
NamesTrain = []
for i in imageNames:
   NamesTrain.append(i.split("\\")[2].split("_")[0])
des_list = get_All_Descriptors(trains)
data = stackDescriptors(des_list)
qdes = query_Descriptors(queries)
kmeans = KMeans(n_clusters=500, random_state=0).fit(data)
histogram = []
for i in range(len(trains)):
    sift = cv2.xfeatures2d.SIFT_create()
    kp,des = sift.detectAndCompute(trains[i],None)
    histogram.append(build_histogram(des,kmeans).tolist())
histogramQuery = []
for i in range(len(queries)):
    sift = cv2.xfeatures2d.SIFT_create()
    kp,des = sift.detectAndCompute(queries[i],None)
    histogramQuery.append(build_histogram(des,kmeans).tolist())
neighQ = []
for i in range(len(histogram)):
    histogram[i].append(NamesTrain[i])
for j in range(len(histogramQuery)):
    histogramQuery[j].append(queryNames[j])
for i in range(len(histogramQuery)):
        neighQ += getNeighbors(histogram,histogramQuery[i],1)
neighbours = []
for i in range(len(histogramQuery)):
        neighbours.append(getNeighbors(histogram,histogramQuery[i],5))
get_Class_Based_Accuracy(histogram,histogramQuery)






























