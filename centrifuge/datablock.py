
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from scipy.stats import entropy


def to_byte_dict(data):
    byte_dict = {}
    for i in range(0, 256): byte_dict.update({i:0})
    for i in list(data): byte_dict[i]+= 1
    return byte_dict


def count_ascii(byte_dict):
    num_ascii = 0
    for i in range(0, 256):
        if byte_dict[i] >= 32 and byte_dict[i] <= 126:
            num_ascii += byte_dict[i]

    return num_ascii


class DataBlock:
    def __init__(self, pathname, data_block, block_size, file_offset):
        # self explanatory
        self.path = pathname
        self.data = data_block
        self.size = block_size
        self.offset = file_offset

        # really annoying. Have to do this if we want to use an externally declared function
        # there must be a better way to do this
        self.to_byte_dict = to_byte_dict
        self.byte_dict = to_byte_dict(self.data)

        # features engineered for this data
        self.entropy = entropy(list(self.byte_dict.values()), base=2)       # entropy

        self.zeroes_ratio = self.byte_dict[0] / block_size                  # percent of bytes that are 0
        self.ascii_ratio = count_ascii(self.byte_dict) / block_size         # percent of bytes that fall within the ASCII range

        self.byteval_std_dev_counts = np.std(list(self.byte_dict.values()))
        self.byteval_std_dev = np.std(list(data_block))

        self.byteval_mean = np.mean(list(data_block))
        self.byteval_median = np.median(list(data_block))

    # methods

    def plot_relative_frequency_distribution(self):

        # unvariate
        #plt.rcParams['figure.figsize'] = [15, 5]
        ax = sns.distplot(np.array(list(self.data)), bins=256, kde=False, norm_hist=True, color='purple');
        ax.set(xlabel='Byte Value (base 10)',
               ylabel='Frequency',
               title='Byte Value Distribution at offset ' + str(self.offset) + ' in ' + self.path)
        # control x axis range
        ax.set_xlim(-10, 260)
        #ax.set_ylim(0, 0.10)
        plt.show()


    def plot_cdf(self):
        #plt.rcParams['figure.figsize'] = [15, 5]
        ax = sns.distplot(np.array(list(self.data)),
                          bins=256,
                          kde=False,
                          hist_kws={'histtype':'step', 'cumulative': True, 'linewidth':1, 'alpha':1}, 
                          kde_kws={'cumulative': True},
                          norm_hist=True,
                          color='red');
        ax.set(xlabel='Byte Value (base 10)',
               ylabel='Probability',
               title='CDF of byte values at offset ' + str(self.offset) + ' in ' + self.path)
        # control x axis range
        ax.set_xlim(-10, 260)
        #ax.set_ylim(0, 0.10)

        plt.show()

