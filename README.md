# Centrifuge

Centrifuge simplifies the application of statistical and machine learning techniques to the analysis of information in binary files. 

This tool demonstrates the utility of two new approaches:

1. An unsupervised machine learning algorithm called DBSCAN is used to cluster byte sequences together based on their statistical properties (features). This is useful because byte sequences belonging to the same data type, e.g. machine code, typically cluster together. As a result, clusters are often representative of a specific data type. Each cluster can be extracted and further analysed. 

2. The specific data type of a cluster found by DBSCAN is identified without machine learning by comparing its byte value distribution to a data type reference distribution by measuring the Wasserstein distance between them. If the Wasserstein distance of the byte value distribution of a cluster is less than a set threshold distance for a particular data type, that cluster will be identified as that data type. 




This toolkit is complements existing tools such as [binwalk](https://github.com/ReFirmLabs/binwalk) and [ISAdetect](https://github.com/kairis/isadetect).
