import os
import json
import pickle
import requests
import matplotlib.pyplot as plt
from scipy.stats import entropy, gamma, relfreq, wasserstein_distance
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns; sns.set()
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from math import ceil, sqrt

from centrifuge.datablock import DataBlock




def find_optimal_eps(matrix, k, epsilon):
        """
        Uses kNN distances to plot a curve. This curve can then be used
        to choose an optimal value of eps for DBSCAN
        """
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(matrix)
        distances, indices = nbrs.kneighbors(matrix)
        sorted_distances = np.sort(np.concatenate(distances[:, -1:]))

        plt.axhline(y=epsilon, color='red')
        plt.text(0, epsilon+0.25, "eps = " + str(epsilon))

        plt.plot(np.arange(len(sorted_distances)), sorted_distances)
        plt.title("K-nearest neighbor distances. Use this plot to choose an optimal epsilon value")
        plt.xlabel("index")
        plt.ylabel("kNN distances for k = " + str(k))
        plt.ylim(0)
        plt.show()




class BinFile:
    def __init__(self, file_handle):
        self.file_handle = file_handle
        self.pathname = self.file_handle.name
        self.block_size = 1024                 # default; will be scaled based on file size, and may be updated manually

        self.file_handle.seek(0)
        #self.data = self.file_handle.read()    # read full file into memory
        self.file_handle.seek(0,2)
        self.size = self.file_handle.tell()
        self.file_handle.seek(0)

        self.debug_level = 0      # default; may be overridden via set_debug_level()

        #
        self.blocks = []
        #self.block_offsets = []
        self.block_offsets = None

        self.block_entropy_levels = None             # uses log base 2, not log base 10
        self.block_zeroes_ratios = None              # % bytes in chunk that are 0
        self.block_ascii_ratios = None               # % bytes in chunk that fall between 32 and 126 inclusive
        self.block_byteval_std_dev = None            #
        self.block_byteval_std_dev_counts = None     # std dev of the counts of each byte value, not the values themselves
        self.block_byteval_mean = None               #
        self.block_byteval_median = None             #
        self.file_data_frame = None                  # data frame built from np arrays of file chunk information stats

        # DBSCAN results stored in this variable
        self.db = None                               # refers to DBSCAN output
        self.dbscan_data_frame = None                # file data frame + cluster labels

    ###############################
    # methods


    def seek(self, offset):
        self.file_handle.seek(offset)


    # manually set block size
    def set_block_size(self, num_bytes):
        self.block_size = num_bytes

    # manually set size of file
    def set_size(self, num_bytes):
        self.size = num_bytes


    def slice_file(self):
        # number of blocks = file size / block size

        # initialize np arrays
        self.block_offsets = np.empty(ceil(self.size / self.block_size), dtype='int64')
        self.block_entropy_levels = np.empty(ceil(self.size / self.block_size), dtype='float64')
        self.block_zeroes_ratios = np.empty(ceil(self.size / self.block_size), dtype='float64')
        self.block_ascii_ratios = np.empty(ceil(self.size / self.block_size), dtype='float64')
        self.block_byteval_std_dev = np.empty(ceil(self.size / self.block_size), dtype='float64')
        self.block_byteval_std_dev_counts = np.empty(ceil(self.size / self.block_size), dtype='float64')
        self.block_byteval_mean = np.empty(ceil(self.size / self.block_size), dtype='float64')
        self.block_byteval_median = np.empty(ceil(self.size / self.block_size), dtype='int64')

        offset = 0    # tracks block offsets
        for i in range(0, (ceil(self.size / self.block_size))):
            new_block = DataBlock(self.pathname,
                                  self.file_handle.read(self.block_size),
                                  self.block_size,
                                  offset)
            self.blocks.append(new_block)

            self.block_offsets[i] = new_block.offset
            self.block_entropy_levels[i] = new_block.entropy
            self.block_zeroes_ratios[i] = new_block.zeroes_ratio
            self.block_ascii_ratios[i] = new_block.ascii_ratio
            self.block_byteval_std_dev[i] = new_block.byteval_std_dev
            self.block_byteval_std_dev_counts[i] = new_block.byteval_std_dev_counts
            self.block_byteval_mean[i] = new_block.byteval_mean
            self.block_byteval_median[i] = new_block.byteval_median

            offset += self.block_size

        # now that the np arrays have been created, build data frame
        self.create_data_frame()


    # should create a dataframe out of these arrays
    def create_data_frame(self):
        '''create data frame from lists'''
        if self.debug_level > 0:
            print("[+] creating data frame")

        self.file_data_frame = pd.DataFrame({'entropy': self.block_entropy_levels,
                                             'zeroes ratios': self.block_zeroes_ratios,
                                             'ascii ratios': self.block_ascii_ratios,
                                             'byte value std dev': self.block_byteval_std_dev,
                                             'byte value counts std dev': self.block_byteval_std_dev_counts,
                                             'byte value mean': self.block_byteval_mean,
                                             'byte value median': self.block_byteval_median})


    def show_scatter_matrix(self):
        '''plots all columns against each other'''
        pd.plotting.scatter_matrix(self.file_data_frame, alpha=0.3, figsize=(20,20), diagonal='kde')
        plt.show()




    # TODO: add docstring
    #def entropy_vs_zeroes_ratios_quickplot(self):
    #
    #    plt.scatter(self.block_zeroes_ratios,
    #                self.block_entropy_levels,
    #                alpha=0.15,
    #                color='purple')
    #
    #    plt.title('Entropy vs. Ratio of 0x00 Byte Values')
    #    plt.xlabel('0x00 Byte Ratio')
    #    plt.ylabel('Entropy')
    #    plt.xlim(-0.05, 1)
    #    plt.ylim(0, 8.5)
    #
    #    plt.show()



    def plot_variables_by_range(self, x, y, start, end, target_data_marker=None, other_data_marker=None, target_data_color=None, other_data_color=None, title=None, xlabel=None, ylabel=None):
        plt.title('test')

        within_range_mask = np.logical_and(self.block_offsets >= start, self.block_offsets <= end)
        out_of_range_mask = np.logical_xor(self.block_offsets >= start, self.block_offsets <= end)

        offsets_within_range = self.block_offsets[within_range_mask]
        x_within_range = x[within_range_mask]
        y_within_range = y[within_range_mask]

        if target_data_marker is None:
            target_data_marker = 's'

        if other_data_marker is None:
            other_data_marker = 'o'

        if target_data_color is None:
            target_data_color = 'red'

        if other_data_color is None:
            other_data_color = 'black'

        plt.plot(x_within_range, y_within_range, target_data_marker, color = target_data_color, alpha=0.3)

        offsets_out_of_range = self.block_offsets[out_of_range_mask]
        x_out_of_range = x[out_of_range_mask]
        y_out_of_range = y[out_of_range_mask]

        plt.plot(x_out_of_range, y_out_of_range, other_data_marker, color = other_data_color, alpha=0.3)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.show()




    def plot_file_entropy(self, start=None, end=None):
        '''
        start and none are numbers representing offsets within the file.
        Can be decimal or hexadecimal.
        '''

        try:
            plt.axvline(x=start, color='red')
            plt.axvline(x=end, color='red')
        except TypeError:
            pass

        plt.plot(self.block_offsets,
                 self.block_entropy_levels,
                 linewidth=0.8,
                 color='blue')

        plt.title("Entropy of " + self.pathname.split('/')[-1:][0])
        plt.xlabel('Offset')
        plt.ylabel('Entropy')
        plt.ylim(-0.5, 8.25)

        plt.show()


    def plot_file_feature(self, feature, color=None, start=None, end=None):
        '''
        generalized version of plot_file_entropy
        '''
        try:
            plt.axvline(x=start, color='red')
            plt.axvline(x=end, color='red')
        except TypeError:
            pass


        features = {"mean":self.block_byteval_mean,
                    "median":self.block_byteval_median,
                    "std_dev":self.block_byteval_std_dev,
                    "std_dev_counts":self.block_byteval_std_dev_counts,
                    "entropy":self.block_entropy_levels,
                    "ascii":self.block_ascii_ratios,
                    "zeroes":self.block_zeroes_ratios}

        if color is None:
            color = "black"

        if feature in features:
            plt.plot(self.block_offsets,
                     features[feature],
                     linewidth=1.1,
                     color=color)
        else:
            print("Feature argument must be one of the following: ")
            for feature in features:
                print(feature)

        plt.title(self.pathname.split('/')[-1:][0])
        plt.xlabel('Offset')
        plt.ylabel(feature)
        plt.show()


    #
    def set_debug_level(self, level):
        self.debug_level = level

        if self.debug_level > 0:
            print("[+]\tDebug level set to %d" % self.debug_level)


    #####################################################################################
    #  Clustering with DBSCAN
    #####################################################################################

    # eps=0.4 and min_sample=10 perform well in general, but
    # eps needs to be increased to 0.7 or higher for files smaller than ~100KB
    # min_sample needs to be increased to 20, 30 or higher for larger (~3MB+) files
    # try finding optimal value of eps using kNN distances

    def cluster_DBSCAN(self, epsilon, minimum_samples, find_optimal_epsilon=True):
        """
        return a data frame containing data from clustering results and the data blocks
        """

        X = StandardScaler().fit_transform(self.file_data_frame) # standardize and scale data frame. Using scikit-learn nc

        if (find_optimal_epsilon==True):
            find_optimal_eps(X, minimum_samples, epsilon)

        self.db = DBSCAN(eps=epsilon, min_samples=minimum_samples).fit(X)
        self.db.n_clusters_ = len(set(self.db.labels_)) - (1 if -1 in self.db.labels_ else 0)

        if self.debug_level > 0:
            print("Set of clusters found by DBSCAN: " + str(set(self.db.labels_)))

        core_samples_mask = np.zeros_like(self.db.labels_, dtype=bool)
        core_samples_mask[self.db.core_sample_indices_] = True

        # create data frame
        db_data_frame = self.file_data_frame.copy(deep=True)
        db_data_frame['core samples mask'] = core_samples_mask
        db_data_frame['cluster labels'] = self.db.labels_

        #return db_data_frame
        self.dbscan_data_frame = db_data_frame
        #
        #    labels = self.db.labels_
        #    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        #
        #    print("Number of clusters found via DBSCAN: " + str(n_clusters_))

        #def cluster_DBSCAN(self, epsilon, minimum_samples):
        #    """
        #    Returns DBSCAN object
        #    """
        #    self.db = ClusterDBSCAN(epsilon, minimum_samples, self.file_data_frame)


    # TODO: outliers (cluster -1) need to be black
    def plot_DBSCAN_results(self):

        """
        Refactored.
        """

        #grid = plt.GridSpec(2, 4, wspace=0.4, hspace=0.3) # <--------------
        #plt.subplot(grid[0, 0:])
        #plt.scatter(np.random.random(100), np.random.random(100))
        #plt.subplot(grid[1, 2:])
        #plt.subplot(grid[1, :2]);

        # rainbow_r, prism, Spectral
        #plt.subplot(grid[0, 0:]) # <-----------
        #plt.plot(self.block_offsets,
        #         self.block_entropy_levels,
        #         linewidth=0.3,
        #         color='black')
        #plt.ylim(-0.25, 8.5)

        labels = self.db.labels_
        unique_labels = set(labels)
        colors = [plt.cm.rainbow_r(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        shapes = ['H','D', 's', 'd', 'o', 'v', 'p', 'h', '^', '>', '<', '.']

        cluster_dfs, _ = self.extract_clusters()

        if cluster_dfs is None:
            print("[!] No clusters to plot. Exiting.")
            return

        for cluster_id in sorted(cluster_dfs.keys()):
            if cluster_id == -1:
                color = "black"
            else:
                color = colors[cluster_id]

            plt.scatter(list(cluster_dfs[cluster_id]["entropy"].index * self.block_size),
                        cluster_dfs[cluster_id]["entropy"],
                        edgecolors="k",
                        marker=shapes[cluster_id],
                        color=color,
                        alpha=1)

        plt.ylim(-0.25, 8.25)
        plt.title(self.pathname.split('/')[-1:][0])
        plt.xlabel("Block Offset")
        plt.ylabel("Block Entropy")
        plt.show()



        for cluster_id in sorted(cluster_dfs.keys()):
            if cluster_id == -1:
                color = "black"
            else:
                color = colors[cluster_id]

            plt.scatter(cluster_dfs[cluster_id]["byte value std dev"],
                        cluster_dfs[cluster_id]["entropy"],
                        edgecolors="k",
                        marker=shapes[cluster_id],
                        color=color,
                        alpha=1)

        plt.xlim(-5)
        plt.ylim(-0.25, 8.25)
        plt.title(self.pathname.split('/')[-1:][0])
        plt.xlabel("Block Byte Value Standard Deviation")
        plt.ylabel("Block Entropy")
        plt.show()



        for cluster_id in sorted(cluster_dfs.keys()):
            if cluster_id == -1:
                color = "black"
            else:
                color = colors[cluster_id]

            plt.scatter(cluster_dfs[cluster_id]["byte value median"],
                        cluster_dfs[cluster_id]["zeroes ratios"],
                        edgecolors="k",
                        marker=shapes[cluster_id],
                        color=color,
                        alpha=1)

        plt.xlabel("Block Printable ASCII Ratio")
        plt.title(self.pathname.split('/')[-1:][0])
        plt.xlabel("Block Median")
        plt.ylabel("Block Zeroes Ratio")
        plt.show()



    def plot_two_features_with_cluster_labels(self, feature_1, feature_2, with_noise=True):
        if feature_1 not in self.dbscan_data_frame.columns or feature_2 not in self.dbscan_data_frame.columns:
            print("Arguments must be 2 of the following: ")
            for feature in self.dbscan_data_frame.columns[0:-2]:
                print(feature)
            return

        labels = self.db.labels_
        unique_labels = set(labels)
        colors = [plt.cm.rainbow_r(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        shapes = ['H','D', 's', 'd', 'o', 'v', 'p', 'h', '^', '>', '<', '.']

        cluster_dfs, _ = self.extract_clusters()
        if with_noise is False:
            cluster_dfs.pop(-1)

        if cluster_dfs is None:
            print("[!] No clusters to plot. Exiting.")
            return


        for cluster_id in sorted(cluster_dfs.keys()):
            if cluster_id == -1:
                color = "black"
            else:
                color = colors[cluster_id]

            plt.scatter(cluster_dfs[cluster_id][feature_1],
                        cluster_dfs[cluster_id][feature_2],
                        edgecolors="k",
                        marker=shapes[cluster_id],
                        color=color,
                        alpha=1)

        plt.xlabel(feature_1)
        plt.ylabel(feature_2)
        plt.title(self.pathname.split('/')[-1:][0] + " clusters")
        plt.show()




    def extract_clusters(self):
        cluster_dataframes = {} # key = cluster ID, value = that cluster's data frame
        cluster_bytes = {}      # key = cluster ID, value = list of all bytes in cluster

        if self.dbscan_data_frame is not None:
            cluster_labels = list(set(self.dbscan_data_frame["cluster labels"])) # example output: [0, 1, 2, -1]
            for label in cluster_labels:

                # extract data frame
                cluster_df = self.dbscan_data_frame[self.dbscan_data_frame["cluster labels"] == label]
                cluster_dataframes[label] = cluster_df

                # extract data/bytes of all blocks in cluster
                bytes = []
                blocks = [self.blocks[i] for i in cluster_df.index]

                for block in blocks:
                    bytes += block.data

                cluster_bytes[label] = bytes

        else:
            print("[!] No cluster data frames to extract\n")
            return None, None

        return cluster_dataframes, cluster_bytes




    def load_data_type_distributions(self):
        distributions = {}
        base_directory = os.path.dirname(__file__)
        load_path = base_directory + "/distributions/data_types/"

        for file in os.listdir(load_path):
            if os.path.isdir(load_path + file):
                continue
            with open(load_path + file, "rb") as f:
                try:
                    distributions[file] = pickle.load(f)
                except:
                    continue

        return distributions



    def load_machine_code_distributions(self):
        distributions = {}
        base_directory = os.path.dirname(__file__)
        load_path = base_directory + "/distributions/cpu_architectures/"

        for file in os.listdir(load_path):
            if os.path.isdir(load_path + file):
                continue
            with open(load_path + file, "rb") as f:
                try:
                    distributions[file] = pickle.load(f)
                except:
                    continue

        return distributions



    def id_code_clusters(self, cluster_dfs, cluster_bytes, reference_dist):

        distances = {}  # store initial distance measurements between clusters and data type distributions
        closely_matching_arch_ref = False
        arch_classification = None
        for id, bytes in cluster_bytes.items():
            id_string = "Cluster " + str(id)
            initial_d = wasserstein_distance(reference_dist, bytes)
            distances[id_string] = initial_d
            in_code_range = False
            if (cluster_dfs[id]["entropy"].mean() > 5.2 and cluster_dfs[id]["entropy"].mean() < 6.8): # Initial cutoff.
                in_code_range = True
                arch_distances = {}  # store distance measurements between a cluster and each CPU arch. ref. dist.
                mc_reference_distributions = self.load_machine_code_distributions()
                print("[+] Checking %s for possible match" % id_string)
                for arch, ref_bytes in mc_reference_distributions.items():
                    code_d = wasserstein_distance(ref_bytes, bytes)
                    arch_distances[arch] = code_d
                    if code_d <= 10: # second cutoff. Looking for close matches
                        closely_matching_arch_ref = True

                if closely_matching_arch_ref is True:
                    print("[+] Closely matching CPU architecture reference(s) found for %s" % id_string)
                    arch_classification = self.get_arch_ID(bytes)
                else:
                    if in_code_range is True:
                        print("[X] No closely matching CPU architecture reference found.\n\n")

                distances[id_string] = [distances[id_string], arch_distances]
                closely_matching_arch_ref = False

        if arch_classification is None:
            print("[X] No machine code cluster detected\n\n")

        #distances = json.dumps(distances, indent = 4)
        return distances, arch_classification




    def id_utf8_en_clusters(self, cluster_bytes, reference_dist):
        distances = {}
        match_found = False
        for id, bytes in cluster_bytes.items():
            id_string = "Cluster " + str(id)
            d =  wasserstein_distance(reference_dist, bytes)
            distances[id_string] = d
            if d < 30: # initial cutoff
                print("[+] UTF-8 (english) detected in %s\n    Wasserstein distance to reference: %s\n" % (id_string, d))
                match_found = True

        if match_found is False:
            print("[X] No UTF-8 (english) cluster detected.\n")

        #distances = json.dumps(distances, indent = 4)
        #print(distances)
        return distances



    def id_high_entropy_clusters(self, cluster_dfs, cluster_bytes, reference_dist):
        distances = {}
        match_found = False
        for id, bytes in cluster_bytes.items():
            id_string = "Cluster " + str(id)
            d =  wasserstein_distance(reference_dist, bytes)
            distances[id_string] = d
            if d < 10: # initial cutoff
                print("[+] High entropy data found in %s\n    Wasserstein distance to reference: %s" % (id_string, d))
                match_found = True
                if d < 1:
                    print("[*] This distance suggests the data in this cluster is either\n    a) encrypted\n" \
                          "    b) compressed via LZMA with maximum compression level.\n")
                else:
                    print("[*] This distance suggests the data in this cluster is compressed\n")

        if match_found is False:
            print("[X] No high entropy data cluster detected.\n")

        #distances = json.dumps(distances, indent = 4)
        #print(distances)
        return distances




    def identify_cluster_data_types(self, show_all=False):
        cluster_dfs, cluster_bytes = self.extract_clusters()
        cluster_bytes.pop(-1, None)                                       # get rid of noise
        reference_distributions = self.load_data_type_distributions()

        print("Searching for machine code\n--------------------------------------------------------------------\n")
        code_distances, arch_classification = self.id_code_clusters(cluster_dfs, cluster_bytes, reference_distributions["machine_code"])

        print("\nSearching for utf8-english data\n-------------------------------------------------------------------\n")
        utf8_en_distances = self.id_utf8_en_clusters(cluster_bytes, reference_distributions["utf8_english"])

        print("\nSearching for high entropy data\n-------------------------------------------------------------------\n")
        high_entropy_distances = self.id_high_entropy_clusters(cluster_dfs, cluster_bytes, reference_distributions["max_entropy"])

        full_results = {"machine code": [code_distances, arch_classification],
                        "utf8_en": utf8_en_distances,
                        "high entropy": high_entropy_distances}

        if show_all is True:
            print("\n\nFull results: \n")
            print(json.dumps(full_results, indent=4))

        return full_results




    def get_arch_ID(self, data):
        print("[+] Sending sample to https://isadetect.com/")
        req = requests.post("https://isadetect.com/binary/",
                            files = { "binary":bytes(data) },
                            data = {"type": "code"})
        print("[+] response:\n")
        response = json.dumps(req.json(), indent=4, sort_keys=True)
        print(response + "\n")

        return req.json()



    def plot_cluster_cdfs(self):
        #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        #          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        counter = 0

        _, cluster_bytes = self.extract_clusters()
        cluster_bytes.pop(-1, None)
        for cluster_id, bytes in cluster_bytes.items():
            sns.distplot(bytes,
                         norm_hist=True,
                         kde=False,
                         hist_kws={'histtype':'step', 'cumulative': True, 'linewidth':2, 'alpha':1},
                         kde_kws={'cumulative': True},
                         bins=256,
                         color=colors[counter % len(colors)]) # wrap around
            counter += 1
            plt.title("CDF of Cluster %d" % cluster_id)
            plt.xlim(-10, 265)
            plt.show()



    def plot_cluster_histograms(self):
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        counter = 0

        _, cluster_bytes = self.extract_clusters()
        cluster_bytes.pop(-1, None)
        for cluster_id, bytes in cluster_bytes.items():
            sns.distplot(bytes,
                         kde=False,
                         hist_kws={'alpha':1},
                         bins=256,
                         color=colors[counter % len(colors)]) # wrap around
            counter += 1
            plt.title("Byte Value Histogram of Cluster %d" % cluster_id)
            plt.xlim(-10, 265)
            plt.show()



    def cluster_scatterplot_matrix(self):
        sns.pairplot(self.dbscan_data_frame[self.dbscan_data_frame["cluster labels"] != -1].drop(["core samples mask"], axis=1), hue="cluster labels")


    def violinplot_cluster_by_feature(self, feature):
        if feature not in self.dbscan_data_frame.columns:
            print("[!] feature must be one of the following:\n")
            for column in self.dbscan_data_frame.drop(["core samples mask", "cluster labels"], axis=1).columns:
                print(column)
            return

        with sns.axes_style("whitegrid"):
            sns.color_palette("rainbow")
            sns.violinplot(x = "cluster labels",
                           y = feature,
                           data = self.dbscan_data_frame[self.dbscan_data_frame["cluster labels"] != -1])
        plt.title(self.pathname.split('/')[-1:][0] + " clusters")
        plt.show()



    def boxplot_cluster_by_feature(self, feature):
        if feature not in self.dbscan_data_frame.columns:
            print("[!] feature must be one of the following:\n")
            for column in self.dbscan_data_frame.drop(["core samples mask", "cluster labels"], axis=1).columns:
                print(column)
            return

        with sns.axes_style("whitegrid"):
            sns.boxplot(x = "cluster labels",
                        y = feature,
                        data = self.dbscan_data_frame[self.dbscan_data_frame["cluster labels"] != -1])
            plt.title(self.pathname.split('/')[-1:][0] + " clusters")
            plt.show()

