from inspect import Parameter
import sys
import json
import glob


class hmmlearn:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.dictionary = []
        self.tags = []
        self.countTags = dict()
        self.open_class = []
        self.emission = dict()
        self.transition = dict()
        self.model = dict()

    def data_split(self, unit: str):
        length = len(unit)
        word = ""
        tag = ""
        for i in range(length):
            if unit[-(i + 1)] == '/':
                word = unit[:length - i - 1]
                tag = unit[length - i:]
                break
        return word, tag

    def countTrans(self, trans_tags: list):
        for i in range(len(trans_tags) - 1):
            prev_tag = trans_tags[i]
            curr_tag = trans_tags[i + 1]
            if prev_tag not in self.transition:
                self.transition[prev_tag] = {}
            if curr_tag in self.transition[prev_tag]:
                self.transition[prev_tag][curr_tag] += 1
            else:
                self.transition[prev_tag][curr_tag] = 1

    def probabilities(self, dict: dict()):
        for tag in dict:
            total = 0
            for num in dict[tag].values():
                total += num
            for key, val in dict[tag].items():
                dict[tag][key] = val / total

    def smoothing(self, dict: dict()):
        for tag1 in dict:
            for target in self.tags:
                if target not in dict[tag1]:
                    dict[tag1][target] = 1

    def hmmModel(self, file: str):
        # use utf8
        text = open(file, mode='r', encoding='utf8')
        # with open(file, mode='r', encoding='utf8') as text:
        #     json.dump(parameter, )
        lines = text.read().split("\n")
        text.close()
        for line in lines:
            # trans_tags = abstractTags(line)
            # countTrans(trans_tags)
            trans_tags = []
            units = line.split()
            for unit in units:
                word, tag = self.data_split(unit)  # count for emission
                if word not in self.dictionary:
                    self.dictionary.append(word)
                if tag not in self.tags:
                    self.tags.append(tag)

                # if tag not in self.countTags:
                #     self.countTags[tag] = 1
                # else:
                #     self.countTags[tag] += 1

                trans_tags.append(tag)

                if tag not in self.emission:
                    self.emission[tag] = {}

                if word in self.emission[tag]:
                    self.emission[tag][word] += 1
                    self.countTags[tag] += 1
                else:
                    self.emission[tag][word] = 1
                    self.countTags[tag] = 1
            # count for transition
            self.countTrans(trans_tags)
        self.probabilities(self.emission)
        # print("test")
        # for item in self.transition:
        #     print(item)
        self.smoothing(self.transition)
        self.probabilities(self.transition)
        # self.open_class = sorted(
        #     self.countTags, key=self.countTags.get, reverse=True)[:1]
        self.open_class = sorted(
            self.countTags, key=self.countTags.get, reverse=True)[:5]

    def createModel(self):
        self.hmmModel(self.file_path)
        # 'Tags': self.tags,
        self.model = {'Dictionary': self.dictionary, 'Tags': self.tags, 'Open_class': self.open_class,
                      'Emission': self.emission, 'Transition': self.transition}
        with open('hmmmodel.txt', 'w') as data:
            json.dump(self.model, data)


path = hmmlearn(sys.argv[1])
path.createModel()
