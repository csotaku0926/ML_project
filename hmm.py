import os
import subprocess
from tqdm import tqdm

class HMM:
    def __init__(self, k=0.1, 
                 outdir="../output/", outfile="dev.p1.out"):
        self.emission_probs = dict()
        self.transition_probs = dict()

        self.k = k
        self.outdir = outdir
        self.outfile = outfile

    def _get_emission(self, y, x, fraction=False):
        """
        return emission parameter using MLE:
        e(x|y) = $frac{count(y->x)}{count(y) + k}$ # x is in dict
               = $frac{k}{count(y) + k}$           # x is not in dict
        ---
        input:
        - y: token
        - x: natural word
        - fraction: if the output is divided by sum
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
    
    def _get_transition(self, y_prev, y_next, fraction=False):
        """
        return transition parameter using MLE:
        ---
        input:
        - y_prev: previous token (y_{i-1} or START)
        - y_next: next token (y_i or STOP)
        - fraction: if the output is divided by sum
        """
        if (y_prev not in self.transition_probs):
            return 0.0
        
        count_x = 0.0
        if (y_next in self.transition_probs[y_prev]):
            count_x = self.transition_probs[y_prev][y_next]
        
        if (fraction):
            count_y = sum([self.transition_probs[y_prev][_x] for _x in self.transition_probs[y_prev]])
            return count_x / count_y

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
        emission parameter e(x | y) is defined as $frac{count(y->x)}{count(y)}$
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

    def calc_transition(self, train_data:list):
        """
        Part 2-1:
        transition parameter q(y_i | y_{i-1}}) is defined as $frac{Count(y_i-1, y_i)}{Count(y_i)}$
        """
        # special case: q(STOP | y_n) and q(y_1 | START)
        self.transition_probs["START"] = dict()
        y_prev = "START"
        stop = "STOP"
        for data in tqdm(train_data):
            # blank line
            if (len(data) <= 1):
                if (stop not in self.transition_probs[y_prev]):
                    self.transition_probs[y_prev][stop] = 0
                self.transition_probs[y_prev][stop] += 1
                y_prev = "START"
                continue
            
            # we only care tokens this time
            data_pair = data[:-1].split()
            y = data_pair[1]
            if (y_prev not in self.transition_probs):
                self.transition_probs[y_prev] = dict()
            
            if (y not in self.transition_probs[y_prev]):
                self.transition_probs[y_prev][y] = 0

            self.transition_probs[y_prev][y] += 1
            y_prev = y
        
        print("length of unique tokens in train data:", len(self.transition_probs))

    def Viterbi(self, seq:list):
        """
        Part 2-2:
        [ref](https://www.cis.upenn.edu/~cis2620/notes/Example-Viterbi-DNA.pdf)

        The probability of the most probable path ending in state h with observation "A"
        at ith position is (assuming the (i-1)th observation is "C", state 'k's are all other state) :
        $p_h(A, i) = e_h(A) * max_k(p_k(C, i-1) * p_{kl})$

        to handle underflow caused by probability products, we use "log_2" function here

        ---
        input:
        - seq: input sequence, e.g. ["I", "love", "ML"]
        """
        output = []
        dp_probs = dict()

        for word in seq:
            for h in self.emission_probs:
                e_h_word = self._get_emission(h, word)

    
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
        
def get_data(dir:str):
    """
    given a file path, return lines read from it
    """
    if (not os.path.exists(dir)):
        print(f"{dir} not exist")
        return []

    data = []
    with open(dir, "r") as f:
        data = f.readlines()
    return data
    
"""
In this project, states (y_i) are defined as tags, observations (x_i) are defined as natural language words
"""
if __name__ == '__main__':
    # define arguments
    # please prepare your training and validation data under "train_dir"
    train_dir = "../EN"
    outdir, outfile = "../output", "dev.p1.out"
    
    # training
    train_data = get_data(f"{train_dir}/train")

    hmm = HMM(outdir=outdir, outfile=outfile)
    # hmm.calc_emission(train_data)
    hmm.calc_transition(train_data[:28])
    print(hmm.transition_probs)
    print(hmm._get_transition("O", "STOP"))

    # validation
    # dev_data = get_data(f"{train_dir}/dev.in")
    # hmm.predict(dev_data)
    # # run eval/eval.py with subprocess
    # eval_out = subprocess.check_output(["python", 
    #                                     "eval/eval.py", 
    #                                     f"{train_dir}/dev.out", 
    #                                     f"{outdir}/{outfile}"], text=True)
    # print(eval_out)
