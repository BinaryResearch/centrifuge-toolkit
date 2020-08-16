import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

import sys
sys.path[0:0] = ['.', '..']

from centrifuge.binfile import BinFile

def cluster():
    with open("/bin/cat", "rb") as f:
        cat = BinFile(f)
        cat.set_block_size(1024)
        cat.slice_file()
        cat.cluster_DBSCAN(1, 5, find_optimal_epsilon=True)
        cat.plot_DBSCAN_results()

if __name__=="__main__":
    cluster()
