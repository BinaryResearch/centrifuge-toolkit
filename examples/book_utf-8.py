import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

import sys
sys.path[0:0] = ['.', '..']

from centrifuge.binfile import BinFile

def analyze_text_file():
    path = "../files/ebook/"
    with open(path + "plain_text_utf8", "rb") as f:
        text_file = BinFile(f)
        text_file.set_block_size(1024)
        text_file.slice_file()
        text_file.cluster_DBSCAN(1,10, find_optimal_epsilon=True)
        text_file.plot_DBSCAN_results()

if __name__=="__main__":
    analyze_text_file()
