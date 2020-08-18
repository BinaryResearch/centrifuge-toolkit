
#!~/anaconda3/bin/python3


import sys
sys.path[0:0] = ['.', '..']

from centrifuge.binfile import BinFile

def visualize_file_entropy():
    with open("/bin/bash", "rb") as f:
        binfile = BinFile(f)
        binfile.slice_file()
        binfile.plot_file_entropy()

if __name__ == "__main__":
    visualize_file_entropy()
