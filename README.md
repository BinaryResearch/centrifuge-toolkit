# Centrifuge

Centrifuge simplifies the application of statistical and machine learning techniques to the analysis of information in binary files. 

This tool implements two new approaches:

1. An unsupervised machine learning algorithm called DBSCAN is used to cluster byte sequences together based on their statistical properties (features). This is useful because byte sequences belonging to the same data type, e.g. machine code, typically cluster together. As a result, clusters are often representative of a specific data type. Each cluster can be extracted and further analysed. 

2. The specific data type of a cluster can often be identified without using machine learning by measuring the Wasserstein distance between its byte value distribution and a data type reference distribution. If this distance is less than a set threshold for a particular data type, that cluster will be identified as that data type. Currently, reference distributions exist for various CPU architectures, high entropy data, and UTF-8 english.

These two approaches are used together in sequence: first DBSCAN finds clusters, then the Wasserstein distances between cluster's data and the reference distributions are measured to identify their data type.

This toolkit is complements existing tools such as [binwalk](https://github.com/ReFirmLabs/binwalk) and [ISAdetect](https://github.com/kairis/isadetect).
