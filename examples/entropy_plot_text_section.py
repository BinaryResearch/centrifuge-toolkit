#!~/anaconda3/bin/python3


import sys
sys.path[0:0] = ['.', '..']

from centrifuge.binfile import BinFile

def label_text_section():
    with open("/bin/bash", "rb") as f:
        binfile = BinFile(f)
        binfile.set_block_size(1024)
        binfile.slice_file()
        binfile.plot_file_entropy(0x2cbc0, 0x2cbc0 + 0xa2c02)

if __name__ == "__main__":
    label_text_section()

