import sys
sys.path[0:0] = ['.', '..']

from centrifuge.binfile import BinFile

def plot():
    with open("/bin/bash", "rb") as f:
        binfile = BinFile(f)
        binfile.set_block_size(1024)
        binfile.slice_file()

        binfile.plot_variables_by_range(binfile.block_entropy_levels,
                                        binfile.block_byteval_std_dev,
                                        0x2cbc0, 0x2cbc0 + 0xa2c02,
                                        title="bash",
                                        xlabel="entropy",
                                        ylabel="byte value standard deviation")

if __name__ == "__main__":
    plot()
