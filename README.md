# Centrifuge

Centrifuge simplifies the application of statistical and machine learning techniques to the analysis of information in binary files. 

This tool implements two new approaches to file analysis:

1. [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), an unsupervised machine learning algorithm, is used to cluster byte sequences together based on their statistical properties (features). This is useful because byte sequences belonging to the same data type, e.g. machine code, typically cluster together. As a result, clusters are often representative of a specific data type. Each cluster can be extracted and analysed further. 

2. The specific data type of a cluster can often be identified without using machine learning by measuring the [Wasserstein distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html) between its byte value distribution and a data type reference distribution. If this distance is less than a set threshold for a particular data type, that cluster will be identified as that data type. Currently, reference distributions exist for various CPU architectures, high entropy data, and UTF-8 english.

These two approaches are used together in sequence: first DBSCAN finds clusters, then the Wasserstein distances between the clusters' data and the reference distributions are measured to identify their data type.

