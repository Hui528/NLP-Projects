import glob
import sys
import math
import json


class nbclassify:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.model = dict()
        self.result = ''

    def readParams(self):
        with open("nbmodel.txt", 'r') as data:
            self.model = json.load(data)

    def readTestData(self):
        for file in glob.glob(self.file_path + '/**/*.txt', recursive=True):
            if file.split('/')[-1] == "README.txt":
                continue
            self.classifier(file)

    def classifier(self, file: str):
        text = open(file, mode='r', encoding='latin1')
        words = text.read().split()
        text.close()
        prob_in_nega = math.log(self.model["prior_nega_class"])
        prob_in_posi = math.log(self.model["prior_posi_class"])
        prob_in_decep = math.log(self.model["prior_decep_class"])
        prob_in_truth = math.log(self.model["prior_truth_class"])
        wordsAndCounts = dict()
        for word in words:
            word = word.rstrip('.,?!"').lower()
            if word in wordsAndCounts:
                wordsAndCounts[word] += 1
            else:
                wordsAndCounts[word] = 1
        for key, value in wordsAndCounts.items():
            if key in self.model["prob_words_nega"]:
                prob_in_nega += value * self.model["prob_words_nega"][key]
            else:
                prob_in_nega += value * \
                    math.log(
                        1.0 / (self.model["num_words_negative"] + self.model["allWordsSet_length"]))
            if key in self.model["prob_words_posi"]:
                prob_in_posi += value * self.model["prob_words_posi"][key]
            else:
                prob_in_posi += value * \
                    math.log(
                        1.0 / (self.model["num_words_positive"] + self.model["allWordsSet_length"]))
            if key in self.model["prob_words_decep"]:
                prob_in_decep += value * self.model["prob_words_decep"][key]
            else:
                prob_in_decep += value * \
                    math.log(
                        1.0 / (self.model["num_words_deceptive"] + self.model["allWordsSet_length"]))
            if key in self.model["prob_words_truth"]:
                prob_in_truth += value * self.model["prob_words_truth"][key]
            else:
                prob_in_truth += value * \
                    math.log(
                        1.0 / (self.model["num_words_truthful"] + self.model["allWordsSet_length"]))

        self.result += 'deceptive\t' if prob_in_decep > prob_in_truth else 'truthful\t'
        self.result += 'negative\t' if prob_in_nega > prob_in_posi else 'positive\t'
        self.result += file + '\n'

    def writeResults(self):
        self.readParams()
        self.readTestData()
        output = open('nboutput.txt', 'w')
        output.write(self.result)
        output.close()


classification = nbclassify(sys.argv[1])
classification.writeResults()
