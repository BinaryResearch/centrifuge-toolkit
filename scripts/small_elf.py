import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

import sys
sys.path[0:0] = ['.', '..']

from centrifuge.binfile import BinFile

def cluster():
    with open("/bin/cat", "rb") as f:
        cat = BinFile(f)
        cat.slice_file()
        cat.cluster_DBSCAN(1, 3, find_optimal_epsilon=True)
        cat.plot_DBSCAN_results()

        cat.identify_cluster_data_types()

if __name__=="__main__":
    cluster()
