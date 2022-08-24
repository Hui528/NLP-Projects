import glob
import sys
import math
import json


class hmmdecode:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.model = dict()
        self.dictionary = []
        self.tags = []
        self.open_class = []
        self.emission = dict()
        self.transition = dict()
        self.result = ''

    def readParams(self):
        with open("hmmmodel.txt", 'r') as data:
            self.model = json.load(data)
        self.dictionary = self.model['Dictionary']
        self.tags = self.model['Tags']
        self.open_class = self.model['Open_class']
        self.emission = self.model['Emission']
        self.transition = self.model['Transition']

    def addTags(self, line: str):
        reverse_tagged_line = []
        words = line.split()
        if len(words) == 0:
            return []
        obs = []
        prob = dict()
        for word in words:
            obs.append(word)
        for t in range(len(obs)):
            prob[t] = dict()
            curr_word_in_dictionary = obs[t] in self.dictionary
            if t == 0:
                # for first word, only use state (emission)
                for q in self.tags:
                    if obs[t] in self.emission[q]:
                        prob[t][q] = [self.emission[q][obs[t]], '']
                    else:
                        prob[t][q] = [0, '']
                if not curr_word_in_dictionary:
                    for oc_tag in self.open_class:
                        prob[t][oc_tag] = [0.2, '']
                continue
                # for q in self.tags:
                #     if not curr_word_in_dictionary:
                #         for oc_tag in self.open_class:
                #             prob[t][oc_tag] = [0.2, '']
                #     else:
                #         if obs[t] in self.emission[q]:
                #             prob[t][q] = [self.emission[q][obs[t]], '']
                #         else:
                #             prob[t][q] = [0, '']

                # continue

            for tag in self.tags:

                # if 0 == self.emission[tag][obs[t]]:
                # this step combine the situation that obs[t] not exist
                # in the dictionary and obs[t] exist in the dictionary but
                # self.emission[tag][obs[t]] == 0 (so obs[t] didn't be tagged
                # as 'tag')
                # update: if obs[t] exist in dictionary but self.emission[tag][obs[t]] == 0
                if curr_word_in_dictionary and obs[t] not in self.emission[tag]:
                    prob[t][tag] = [0, '']
                else:
                    # IMPORTANT:seems has a problem, need smooth instead of set p = 0 when trans from prev_tag to curr_tag not exist. Otherwise, not fully connected, and the path != 0 not exist
                    p = 0
                    for prev_q in self.tags:
                        curr_p = prob[t - 1][prev_q][0] * \
                            self.transition[prev_q][tag]
                        if curr_p > p:
                            p = curr_p
                            q = prev_q

                    if curr_word_in_dictionary:
                        # q is the previous state that maximize the prob[t][tag]
                        prob[t][tag] = [p * self.emission[tag][obs[t]], q]
                    else:
                        # if tag in self.open_class:
                        #     prob[t][tag] = [p, q]
                        # else:
                        #     prob[t][tag] = [0, q]
                        prob[t][tag] = [p, q]

        last_tag = ""
        last_prob = 0

        for lt in prob[len(obs) - 1]:
            if prob[len(obs) - 1][lt][0] >= last_prob:
                last_prob = prob[len(obs) - 1][lt][0]
                last_tag = lt

        reverse_tagged_line.append(obs[-1] + '/' + last_tag)
        for rev_t in range(len(obs) - 1, 0, -1):
            tag = prob[rev_t][last_tag][1]
            reverse_tagged_line.append(obs[rev_t - 1] + '/' + tag)
            last_tag = tag

        return reverse_tagged_line

    def readData(self):
        # use utf8
        text = open(self.file_path, mode='r', encoding='utf8')
        lines = text.read().split("\n")
        text.close()
        for line in lines:
            wordsNtags = ""
            tagged_line = self.addTags(line)
            if len(tagged_line) != 0:
                for i in range(len(tagged_line) - 1, 0, -1):
                    wordsNtags += tagged_line[i] + ' '
                wordsNtags += tagged_line[0]
                self.result += wordsNtags + '\n'

    def output(self):
        output = open('hmmoutput.txt', 'w')
        output.write(self.result)
        output.close()


decode = hmmdecode(sys.argv[1])
decode.readParams()
decode.readData()
decode.output()
