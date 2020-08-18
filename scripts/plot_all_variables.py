import sys
sys.path[0:0] = ['.', '..']

from centrifuge.binfile import BinFile


def plot_all():
    with open("/bin/bash", "rb") as f:
        binfile = BinFile(f)
        binfile.slice_file()

        binfile.show_scatter_matrix()
        print(binfile.file_data_frame)


if __name__ == "__main__":
    plot_all()

