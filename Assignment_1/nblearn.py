import sys
import math
import json
import glob


class nblearn:
    def __init__(self, file_path: str):
        self.file_path = file_path

        self.num_class = {"negative": 0, "positive": 0,
                          "deceptive": 0, "truthful": 0}

        self.num_words = {"negative": 0, "positive": 0,
                          "deceptive": 0, "truthful": 0}

        self.nega_dict = dict()
        self.posi_dict = dict()
        self.decep_dict = dict()
        self.truth_dict = dict()
        self.allWordsSet = set()

        self.model = dict()

    def readData(self):
        for file in glob.glob(self.file_path + '/**/*.txt', recursive=True):
            if file.split('/')[-4].split('_')[0] == "negative":
                self.num_words["negative"] += self.countWords(
                    file, self.nega_dict)
                self.num_class["negative"] += 1
            if file.split('/')[-4].split('_')[0] == "positive":
                self.num_words["positive"] += self.countWords(
                    file, self.posi_dict)
                self.num_class["positive"] += 1
            if file.split('/')[-3].split('_')[0] == "deceptive":
                self.num_words["deceptive"] += self.countWords(
                    file, self.decep_dict)
                self.num_class["deceptive"] += 1
            if file.split('/')[-3].split('_')[0] == "truthful":
                self.num_words["truthful"] += self.countWords(
                    file, self.truth_dict)
                self.num_class["truthful"] += 1
        self.model["num_words_negative"] = self.num_words["negative"]
        self.model["num_words_positive"] = self.num_words["positive"]
        self.model["num_words_deceptive"] = self.num_words["deceptive"]
        self.model["num_words_truthful"] = self.num_words["truthful"]
        self.model["allWordsSet_length"] = len(self.allWordsSet)

    def countWords(self, file: str, dict: dict()):
        num_words = 0
        text = open(file, mode='r', encoding='latin1')
        words = text.read().split()
        text.close()
        for word in words:
            num_words += 1
            word = word.rstrip('.,?!"').lower()
            self.allWordsSet.add(word)
            if word in dict:
                dict[word] += 1
            else:
                dict[word] = 1
        return num_words

    def probabilities(self):
        self.model['prior_nega_class'] = self.num_class["negative"] / \
            (self.num_class["negative"] + self.num_class["positive"])
        self.model['prior_posi_class'] = self.num_class["positive"] / \
            (self.num_class["negative"] + self.num_class["positive"])
        self.model['prior_decep_class'] = self.num_class["deceptive"] / \
            (self.num_class["deceptive"] + self.num_class["truthful"])
        self.model['prior_truth_class'] = self.num_class["negative"] / \
            (self.num_class["deceptive"] + self.num_class["truthful"])

        wordsSetLen = self.model["allWordsSet_length"]
        self.model['prob_words_nega'] = dict()
        self.model['prob_words_posi'] = dict()
        self.model['prob_words_decep'] = dict()
        self.model['prob_words_truth'] = dict()
        for key, value in self.nega_dict.items():
            self.model['prob_words_nega'][key] = math.log(
                (value + 1) / (self.num_words["negative"] + wordsSetLen))
        for key, value in self.posi_dict.items():
            self.model['prob_words_posi'][key] = math.log(
                (value + 1) / (self.num_words["positive"] + wordsSetLen))
        for key, value in self.decep_dict.items():
            self.model['prob_words_decep'][key] = math.log(
                (value + 1) / (self.num_words["deceptive"] + wordsSetLen))
        for key, value in self.truth_dict.items():
            self.model['prob_words_truth'][key] = math.log(
                (value + 1) / (self.num_words["truthful"] + wordsSetLen))

    def createModel(self):
        self.readData()
        self.probabilities()
        with open('nbmodel.txt', 'w') as data:
            json.dump(self.model, data)


path = nblearn(sys.argv[1])
path.createModel()
