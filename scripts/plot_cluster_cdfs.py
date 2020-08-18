import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

import sys
sys.path[0:0] = ['.', '..']

from centrifuge.binfile import BinFile

def cluster():
    with open("/bin/ls", "rb") as f:
        binfile = BinFile(f)
        binfile.slice_file()
        binfile.cluster_DBSCAN(0.9, 5, find_optimal_epsilon=False)
        #binfile.plot_DBSCAN_results()

        binfile.plot_cluster_cdfs()


if __name__=="__main__":
    cluster()
