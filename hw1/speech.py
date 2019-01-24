#!/bin/python

def read_files(tarfname):

	import tarfile
	tar = tarfile.open(tarfname, "r:gz")

	'''Loading Data into speech'''
	class Data: pass
	speech = Data()
	speech.train_data, speech.train_fnames, speech.train_labels = read_tsv(tar, "train.tsv")
	speech.dev_data, speech.dev_fnames, speech.dev_labels = read_tsv(tar, "dev.tsv")

	'''Stop Words Data'''
	#from nltk.corpus import stopwords
	#stop_words_list = set(stopwords.words('english'))
	stop_words_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

	print("\nTransforming data and labels")
	print("--> Removing stop words")
	from sklearn.feature_extraction.text import CountVectorizer
	speech.count_vect = CountVectorizer(stop_words=stop_words_list)
	speech.trainX = speech.count_vect.fit_transform(speech.train_data)
	speech.devX = speech.count_vect.transform(speech.dev_data)

	from sklearn import preprocessing
	speech.le = preprocessing.LabelEncoder()
	speech.le.fit(speech.train_labels)
	speech.target_labels = speech.le.classes_
	speech.trainy = speech.le.transform(speech.train_labels)
	speech.devy = speech.le.transform(speech.dev_labels)
	tar.close()
	return speech

def read_unlabeled(tarfname, speech):
	#Reads the unlabeled data --> Returned object contains three fields:
		#data: documents, represented as sequence of words
		#fnames: list of filenames, one for each document
		#X: bag of word vector for each document, using the speech.vectorizer

	import tarfile
	tar = tarfile.open(tarfname, "r:gz")

	'''Loading Data into unlabelled'''
	class Data: pass
	unlabeled = Data()
	unlabeled.data = []
	unlabeled.fnames = []
	for m in tar.getmembers():
		if "unlabeled" in m.name and ".txt" in m.name:
			unlabeled.fnames.append(m.name)
			unlabeled.data.append(read_instance(tar, m.name))
	unlabeled.X = speech.count_vect.transform(unlabeled.data)
	print(unlabeled.X.shape)
	tar.close()
	return unlabeled

def read_tsv(tar, fname):
	member = tar.getmember(fname)
	print("--> {}".format(member.name))
	tf = tar.extractfile(member)
	data = []
	labels = []
	fnames = []
	for line in tf:
		line = line.decode("utf-8")
		(ifname,label) = line.strip().split("\t")
		#print ifname, ":", label
		content = read_instance(tar, ifname)
		labels.append(label)
		fnames.append(ifname)
		data.append(content)
	return data, fnames, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, speech):
	"""Writes the predictions in Kaggle format.

	Given the unlabeled object, classifier, outputfilename, and the speech object,
	this function write the predictions of the classifier on the unlabeled data and
	writes it to the outputfilename. The speech object is required to ensure
	consistent label names.
	"""
	yp = cls.predict(unlabeled.X)
	labels = speech.le.inverse_transform(yp)
	f = open(outfname, 'w')
	f.write("FileIndex,Category\n")
	for i in range(len(unlabeled.fnames)):
		fname = unlabeled.fnames[i]
		# iid = file_to_id(fname)
		f.write(str(i+1))
		f.write(",")
		#f.write(fname)
		#f.write(",")
		f.write(labels[i])
		f.write("\n")
	f.close()

def file_to_id(fname):
	return str(int(fname.replace("unlabeled/","").replace("labeled/","").replace(".txt","")))

def write_gold_kaggle_file(tsvfile, outfname):
	"""Writes the output Kaggle file of the truth.

	You will not be able to run this code, since the tsvfile is not
	accessible to you (it is the test labels).
	"""
	f = open(outfname, 'w')
	f.write("FileIndex,Category\n")
	i = 0
	with open(tsvfile, 'r') as tf:
		for line in tf:
			(ifname,label) = line.strip().split("\t")
			# iid = file_to_id(ifname)
			i += 1
			f.write(str(i))
			f.write(",")
			#f.write(ifname)
			#f.write(",")
			f.write(label)
			f.write("\n")
	f.close()

def write_basic_kaggle_file(tsvfile, outfname):
	"""Writes the output Kaggle file of the naive baseline.

	This baseline predicts OBAMA_PRIMARY2008 for all the instances.
	You will not be able to run this code, since the tsvfile is not
	accessible to you (it is the test labels).
	"""
	f = open(outfname, 'w')
	f.write("FileIndex,Category\n")
	i = 0
	with open(tsvfile, 'r') as tf:
		for line in tf:
			(ifname,label) = line.strip().split("\t")
			i += 1
			f.write(str(i))
			f.write(",")
			f.write("OBAMA_PRIMARY2008")
			f.write("\n")
	f.close()

def read_instance(tar, ifname):
	inst = tar.getmember(ifname)
	ifile = tar.extractfile(inst)
	content = ifile.read().strip()
	return content

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

if __name__ == "__main__":

	# loading, reading data
	print("Reading data")
	tarfname = "speech.tar.gz"
	speech = read_files(tarfname)

	import classify
	print("\nTraining classifier")
	cls = classify.train_classifier(speech.trainX, speech.trainy)

	print("\nEvaluating")
	classify.evaluate(speech.trainX, speech.trainy, cls)
	classify.evaluate(speech.devX, speech.devy, cls)

	#print("\nReading unlabeled data")
	#unlabeled = read_unlabeled(tarfname, speech)

	#print("\nWriting pred file"
	#write_pred_kaggle_file(unlabeled, cls, "speech-pred.csv", speech)

	'''You can't run this since you do not have the true labels'''
	# print "Writing gold file"
	# write_gold_kaggle_file("data/speech-unlabeled.tsv", "data/speech-gold.csv")
	# write_basic_kaggle_file("data/speech-unlabeled.tsv", "data/speech-basic.csv")


