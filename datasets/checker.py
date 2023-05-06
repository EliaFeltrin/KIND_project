import pandas as pd
import os

PATH_TO_DATASET = 'BIO_tag_NER_notation'
nonBIO_PATH = 'Inside_outside_NER_notation'

#function to check if 2 tsv files are equal
def check_equal(file1, file2):
	df1 = pd.read_csv(file1, sep='\t')
	df2 = pd.read_csv(file2, sep='\t')
	#printf the path of the two filr, length of the 2 files, and if they are equal
	print(file1, "\n", 
       	  file2, "\n",
		    "equal: ", df1.equals(df2), "\n\n")


#########################à BIO TAGGER ########################################

oTag = "O"  
types = set()

files = {
	"wikinews_train.tsv"	: "./BIO_tag_NER_notation/automatic/WN_train.tsv",
	"wikinews_test.tsv"		: "./BIO_tag_NER_notation/automatic/WN_dev.tsv",
	"fiction_train.tsv"		: "./BIO_tag_NER_notation/automatic/FIC_train.tsv",
	"fiction_test.tsv"		: "./BIO_tag_NER_notation/automatic/FIC_dev.tsv",
	"degasperi_train.tsv"	: "./BIO_tag_NER_notation/automatic/ADG_train.tsv",
	"degasperi_test.tsv"	: "./BIO_tag_NER_notation/automatic/ADG_dev.tsv",
	"moro_train.tsv"		: "./BIO_tag_NER_notation/automatic/MR_train.tsv",
	"moro_test.tsv"			: "./BIO_tag_NER_notation/automatic/MR_dev.tsv",
}

count = {}

for file in files:
	with open(os.path.join(nonBIO_PATH, file), "r") as f:
		outFile = files[file]
		count[outFile] = {"sentences": 0, "tags": {}, "tokens": 0}

		sentences = []
		thisSentence = []

		for line in f:
			line = line.strip()
			if len(line) == 0:
				if len(thisSentence) > 0:
					sentences.append(thisSentence)
					thisSentence = []
				continue
			parts = line.split("\t")
			thisSentence.append(parts)
			count[outFile]["tokens"] += 1

		if len(thisSentence) > 0:
			sentences.append(thisSentence)

		count[outFile]["sentences"] = len(sentences)

		for sentence in sentences:
			previousNer = oTag
			for token in sentence:
				ner = token[1]
				newNer = ner
				if ner != oTag:
					if previousNer != ner:
						if ner not in count[outFile]["tags"]:
							count[outFile]["tags"][ner] = 0
						newNer = "B-" + ner
						count[outFile]["tags"][ner] += 1
						types.add(ner)
					else:
						newNer = "I-" + ner
				token[1] = newNer
				previousNer = ner

		with open(outFile, "w") as fw:
			for sentence in sentences:
				for token in sentence:
					fw.write(token[0])
					fw.write("\t")
					fw.write(token[1])
					fw.write("\n")
				fw.write("\n")


template = "{:<30} {:>10} {:>10}"
print(template.format("Filename", "Sentences", "Tokens"), end=" ")
for tag in types:
	print("{:>6}".format(tag), end=" ")
print()
for file in count:
	print(template.format(file, count[file]["sentences"], count[file]["tokens"]), end=" ")
	for tag in types:
		print("{:>6}".format(count[file]["tags"][tag]), end=" ")
	print()


#########################à CHECK ########################################
comp = [
	["./BIO_tag_NER_notation/automatic/WN_train.tsv", 	'./' + PATH_TO_DATASET + '/wikinews_train_BIO.tsv'],
	["./BIO_tag_NER_notation/automatic/WN_dev.tsv", 	'./' + PATH_TO_DATASET + '/wikinews_test_BIO.tsv'],
	["./BIO_tag_NER_notation/automatic/FIC_train.tsv", 	'./' + PATH_TO_DATASET + '/fiction_train_BIO.tsv'],
	["./BIO_tag_NER_notation/automatic/FIC_dev.tsv", 	'./' + PATH_TO_DATASET + '/fiction_test_BIO.tsv'],
	["./BIO_tag_NER_notation/automatic/ADG_train.tsv", 	'./' + PATH_TO_DATASET + '/degasperi_train_BIO.tsv'],
	["./BIO_tag_NER_notation/automatic/ADG_dev.tsv", 	'./' + PATH_TO_DATASET + '/degasperi_test_BIO.tsv'],
	["./BIO_tag_NER_notation/automatic/MR_train.tsv", 	'./' + PATH_TO_DATASET + '/moro_train_BIO.tsv'],
	["./BIO_tag_NER_notation/automatic/MR_dev.tsv", 	'./' + PATH_TO_DATASET + '/moro_test_BIO.tsv'],
]

for i in comp:
	check_equal(i[0], i[1])
