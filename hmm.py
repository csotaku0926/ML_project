import os
from tqdm import tqdm

class HMM:
    def __init__(self, k=0.1, 
                 outdir="../output/", outfile="dev.p1.out"):
        self.emission_probs = dict()
        self.k = k
        self.outdir = outdir
        self.outfile = outfile

    def _get_emission(self, y, x, fraction=False):
        """
        return emission parameter using MLE:
        e(x|y) = $frac{count(y->x)}{count(y) + k}$ # x is in dict
               = $frac{k}{count(y) + k}$           # x is not in dict
        """
        if (y not in self.emission_probs):
            return 0.0
        
        count_x = self.k
        if (x in self.emission_probs[y]):
            count_x = self.emission_probs[y][x]
        
        if (fraction):
            count_y = sum([self.emission_probs[y][_x] for _x in self.emission_probs[y]])
            return count_x / (count_y + self.k)

        return count_x

    def _argmax(self, x):
        """
        return the y with the most e(x|y) in self.emission_probs
        """
        ret_y = ''
        prob_y = 0.0
        for y in self.emission_probs:
            prob = self._get_emission(y, x)
            if (prob > prob_y):
                ret_y = y
                prob_y = prob

        return ret_y

    def calc_emission(self, train_data:list):
        """
        Part 1-1: 
        emission parameter is defined as $frac{count(y->x)}{count(y)}$
        if x does not appear in test set, replace count(y->x) with k (occurance of #UNK#)
        ---
        input:
        - train_data: the lines read in train file
        """
        for data in tqdm(train_data):
            # blank line separating sentences
            if (len(data) <= 1):
                continue

            data_pair = data[:-1].split()
            x, y = data_pair[0], data_pair[1]

            if (y not in self.emission_probs):
                self.emission_probs[y] = dict()

            if (x not in self.emission_probs[y]):
                self.emission_probs[y][x] = 0
            
            self.emission_probs[y][x] += 1
        print("length of unique tokens in train data:", len(self.emission_probs))

    def predict(self, test_data:list):
        """
        Part 1-3:
        Output the y* = argmax_y e(x|y) to a file specified by @self.
        based on e(x|y) learned in self.calc_emission
        ---
        input:
        - test_data: list consists of lines of word tokens (format should be same as "dev.in")
        """
        if (not os.path.exists(self.outdir)):
            os.makedirs(self.outdir)

        with open(os.path.join(self.outdir, self.outfile), 'w') as f:
            for token in tqdm(test_data):
                # blank line
                if (len(token) <= 1):
                    f.write("\n")
                    continue
                
                token = token[:-1]
                f.write(token + ' ' + self._argmax(token) + '\n')
        
"""
In this project, states (y_i) are defined as tags, observations (x_i) are defined as natural language words
"""
if __name__ == '__main__':
    train_data = []
    with open("../EN/train") as train_f:
        train_data = train_f.readlines()

    hmm = HMM()
    hmm.calc_emission(train_data)

    with open("../EN/dev.in") as dev_f:
        dev_data = dev_f.readlines()

    hmm.predict(dev_data)
