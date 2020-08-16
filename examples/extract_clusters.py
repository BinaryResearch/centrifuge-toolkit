import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

import sys
sys.path[0:0] = ['.', '..']

from centrifuge.binfile import BinFile


def plot_cluster_CDF(cluster_bytes, title):
    sns.distplot(cluster_bytes, norm_hist=True, kde=False,
                 hist_kws={'histtype':'step', 'cumulative': True, 'linewidth':2, 'alpha':1},
                 kde_kws={'cumulative': True},
                 bins=256)
    plt.title(title)
    plt.show()



def cluster():
    with open("/bin/ls", "rb") as f:
        binfile = BinFile(f)
        binfile.set_block_size(1024)
        binfile.slice_file()
        binfile.cluster_DBSCAN(0.9, 5, find_optimal_epsilon=False)
        binfile.plot_DBSCAN_results()

        cluster_dfs, cluster_bytes = binfile.extract_clusters()

        for cluster_ID, code in cluster_bytes.items():
            plot_cluster_CDF(code, "Cluster " + str(cluster_ID))


if __name__=="__main__":
    cluster()
