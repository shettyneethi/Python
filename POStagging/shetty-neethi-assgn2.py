from math import ceil
from random import shuffle
from sklearn.metrics import accuracy_score
import numpy 

traindata_percent = 80 #percetage of data to be considered for training

data = open('berp-POS-training.txt','r')
data = data.readlines()

#put each line with word details in the list
dataSet = []

for i in data:
	if i != "\n":
		dataSet.append(i.lower().strip().split("\t"))
shuffle(dataSet)
size = len(dataSet)
#print("Total data in   Data   Set = ", size)

training_data_size = ceil(size*traindata_percent/100)
unshuffled_dataSet = list(dataSet)

training_data = dataSet[:training_data_size]
test_data = dataSet[training_data_size:]
test_data_size = len(test_data)

#print("Total data in Training Set = ", training_data_size)
#print("Total data in Testing  Set = ", test_data_size)

#------------------Baseline----------------------
wrd_tag_map = {}
tag_count = {}

#compute frequency in traing data
for i in training_data:
	if i[1] in wrd_tag_map:
		if i[2] in wrd_tag_map[i[1]]:
			wrd_tag_map[i[1]][i[2]] += 1
		else:
			wrd_tag_map[i[1]][i[2]] = 1
	else:
		wrd_tag_map[i[1]] = {i[2] : 1}
	if i[2] in tag_count:
		tag_count[i[2]] += 1
	else:
		tag_count[i[2]]	= 1
	if(i[0]==1):
		try:
			tag_count["<s>"] += 1
		except:
			tag_count["<s>"] = 1
a= tag_count.keys()
#test using test data
pred_labels = []
for i in test_data:
	if(i[1] in wrd_tag_map):
		pred_tag = max(wrd_tag_map[i[1]], key=wrd_tag_map[i[1]].get)
	else:
		pred_tag = max(tag_count, key=tag_count.get)	#handle unknown tags
	pred_labels.append(pred_tag)

true_labels = [l[2] for l in test_data]

#print("Accuracy in percentage = ",accuracy_score(true_labels,pred_labels)*100)

#####################################################################

from math import ceil
from random import shuffle
import numpy
#print("#######------VITERBI------##########")


data_raw = open('berp-POS-training.txt','r')
data_raw = data_raw.readlines()
training_data = []
for i in data_raw:
	if i != "\n":
		training_data.append(i.lower().strip().split("\t"))
training_data_sentences = []
sentence = []
for i in training_data:
	if int(i[0])==1:
		if len(sentence)>0:
			training_data_sentences.append(sentence)
			sentence = []
	sentence.append(i)
	
training_data_sentences.append(sentence)
training_data = training_data_sentences
print(len(training_data))
data_raw = open('assgn2-test-set.txt','r')
data_raw = data_raw.readlines()

test_data = []
for i in data_raw:
	if i != "\n":
		test_data.append(i.lower().strip().split("\t"))
test_data_sentences = []
sentence = []
for i in test_data:
	if int(i[0])==1:
		if len(sentence)>0:
			test_data_sentences.append(sentence)
			sentence = []
	sentence.append(i)
	
test_data_sentences.append(sentence)
test_data = test_data_sentences

unique_words = ["<UNK>"]
unique_tagsWithoutStart = []
for sentence in training_data:
	for word in sentence:
		if word[1] not in unique_words:
			unique_words.append(word[1])
		if word[2] not in unique_tagsWithoutStart:
			unique_tagsWithoutStart.append(word[2])


unique_tagsWithStart = ["<s>"] + unique_tagsWithoutStart

prevTagVsTagMatrix = numpy.zeros((len(unique_tagsWithStart),len(unique_tagsWithoutStart)))
tagVsWordMatrix	= numpy.zeros((len(unique_tagsWithoutStart),len(unique_words)))

for sentence in training_data:
	prev_tag = "<s>"
	for word in sentence:
		prevTagVsTagMatrix[unique_tagsWithStart.index(prev_tag)][unique_tagsWithoutStart.index(word[2])] += 1
		tagVsWordMatrix[unique_tagsWithoutStart.index(word[2])][unique_words.index(word[1])] += 1
		prev_tag = word[2]

#calculate Prob
for i in range(len(unique_tagsWithStart)):
	tagcount = sum(prevTagVsTagMatrix[i])
	for j in range(len(unique_tagsWithoutStart)):
		prevTagVsTagMatrix[i][j] = (prevTagVsTagMatrix[i][j] + (1)) / (tagcount + (len(unique_words)))

for i in range(len(unique_tagsWithoutStart)):
	tagcount = sum(tagVsWordMatrix[i])
	for j in range(len(unique_words)):
		tagVsWordMatrix[i][j] = (tagVsWordMatrix[i][j] + (1)) / (tagcount + (len(unique_words)))

#####################################################################

#######------VITERBI ALGO------##########")

output = ""
for sentence in test_data:
	
	viterbi_matrix = numpy.zeros((len(unique_tagsWithoutStart),len(sentence)))
	backPointers = numpy.zeros((len(unique_tagsWithoutStart),len(sentence)))
	wordList = []
	for word in sentence:
		wordList.append(word[1])

	#INITIALIZATION
	for state in unique_tagsWithoutStart:
		try:
			viterbi_matrix[unique_tagsWithoutStart.index(state)][0] = prevTagVsTagMatrix[0][unique_tagsWithoutStart.index(state)] * tagVsWordMatrix[unique_tagsWithoutStart.index(state)][unique_words.index(wordList[0])]
		except:
			viterbi_matrix[unique_tagsWithoutStart.index(state)][0] = prevTagVsTagMatrix[0][unique_tagsWithoutStart.index(state)] * tagVsWordMatrix[unique_tagsWithoutStart.index(state)][unique_words.index("<UNK>")]
		backPointers[unique_tagsWithoutStart.index(state)][0] = 0
	
	#RECURSION
	for t in range(1,len(sentence)):
		for s in range(len(unique_tagsWithoutStart)):
			max_prob  = 0
			argmax_prob = -1

			for s_dash in range(len(unique_tagsWithoutStart)):
				if wordList[t] in unique_words:
					prob_fromSDash =  viterbi_matrix[s_dash][t-1] * prevTagVsTagMatrix[s_dash+1][s] * tagVsWordMatrix[s][unique_words.index(wordList[t])]
				else:
					prob_fromSDash =  viterbi_matrix[s_dash][t-1] * prevTagVsTagMatrix[s_dash+1][s] * tagVsWordMatrix[s][unique_words.index("<UNK>")]
				if(prob_fromSDash > max_prob):
					max_prob = prob_fromSDash
					argmax_prob = s_dash
			viterbi_matrix[s][t] = max_prob
			backPointers[s][t] = argmax_prob

	#TERMINATION
	bestPathProb = max(viterbi_matrix[:,len(sentence)-1])
	bestPathPointer = int(numpy.argmax(viterbi_matrix[:,len(sentence)-1]))
	
	res = [unique_tagsWithoutStart[bestPathPointer]]
	for column in reversed(range(1,len(sentence))):
		bestPathPointer = int(backPointers[bestPathPointer][column])
		res = [unique_tagsWithoutStart[bestPathPointer]] + res
	s = ""
	for i in range(len(sentence)):
		s += sentence[i][0]+"\t"+sentence[i][1]+"\t"+res[i].upper()+"\n"
	output += (s+"\n")

with open("shetty-neethi-assgn2-test-output.txt","w") as f:
	f.write(output.strip())
