

def plot_file_entropy(bf, start=None, end=None):
    '''bf argument is an instance of the BinFile class, start and none are numbers representing offsets within the file'''

    try:
        plt.axvline(x=start, color='red')
        plt.axvline(x=end, color='red')
    except TypeError:
        pass

    plt.plot(bf.block_offsets,
             bf.block_entropy_levels,
             linewidth=0.8,
             color='blue')

    plt.title("Entropy of " + str(bf.pathname))
    plt.xlabel('Offset')
    plt.ylabel('Entropy')
    plt.ylim(-0.5, 8)

    plt.show()



def quickplot(x, y, c, title, xl, yl):

    plt.scatter(x, y, alpha=0.15, color=c)

    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    #plt.xlim(-0.05, 1)
    #plt.ylim(0, 8.5)

    plt.show()
