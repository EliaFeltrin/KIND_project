"""
This script is used to create the dataset used
in the Evalita 2023 NERMuD task
Link: https://nermud.fbk.eu/
"""

import os

oTag = "O"
types = set()

files = {
	#"wikinews_train.tsv": "wikinews_train_BIO.tsv",
	#"wikinews_test.tsv": "wikinews_test_BIO.tsv",
	#"fiction_train.tsv": "fiction_train_BIO.tsv",
	#"fiction_test.tsv": "fiction_test_BIO.tsv",
	#"degasperi_train.tsv": "degasperi_train_BIO.tsv",
	#"degasperi_test.tsv": "degasperi_test_BIO.tsv",
	"moro_train.tsv": "moro_train_BIO.tsv",
	"moro_test.tsv": "moro_test_BIO.tsv",
}

count = {}

for file in files:
	with open(os.path.join("..", "Inside_outside_NER_notation", file), "r") as f:
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
