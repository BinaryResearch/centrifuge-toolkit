import sys
sys.path[0:0] = ['.', '..']

from centrifuge.binfile import BinFile

def cluster():
    with open("/bin/ls", "rb") as f:
        binfile = BinFile(f)
        binfile.slice_file()
        binfile.cluster_DBSCAN(0.9, 10, find_optimal_epsilon=False)
        results = binfile.identify_cluster_data_types()
        print(results)

if __name__=="__main__":
    cluster()

