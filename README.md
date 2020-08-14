# Centrifuge

Centrifuge simplifies the application of statistical and machine learning techniques to the analysis of information in binary files. 

Two approaches are implemented:

1 . It uses an unsupervised machine learning algorithm called DBSCAN to automatically cluster byte sequences together based on their statistical properties (features). This is useful because byte sequences belonging to the same data type, e.g. machine code or compressed data, typically cluster together. As a result, clusters are often representative of a specific data type. Each cluster can be extracted and further analysed separately. 

2. In addition to this, Centrifuge 




This toolkit is complements existing tools such as [binwalk](https://github.com/ReFirmLabs/binwalk) and [ISAdetect](https://github.com/kairis/isadetect).
