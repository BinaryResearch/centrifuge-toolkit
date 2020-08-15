# Centrifuge

Centrifuge simplifies the application of statistical and machine learning techniques to the analysis of information in binary files. Statistical properties of different types of information can be straightforwardly visualized, measured and objectively quantified.

<hr>

This tool implements two new approaches to file analysis:

1. [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), an unsupervised machine learning algorithm, is used find clusters of byte sequences based on their statistical properties (features). This is useful because byte sequences belonging to the same data type, e.g. machine code, typically cluster together. As a result, clusters are often representative of a specific data type. Each cluster can be extracted and analysed further. 

2. The specific data type of a cluster can often be identified without using machine learning by measuring the [Wasserstein distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html) between its byte value distribution and a data type reference distribution. If this distance is less than a set threshold for a particular data type, that cluster will be identified as that data type. Currently, reference distributions exist for various CPU architectures, high entropy data, and UTF-8 english.

These two approaches are used together in sequence: first DBSCAN finds clusters, then the Wasserstein distances between the clusters' data and the reference distributions are measured to identify their data type.

## Example Use Cases

 - **Determining whether a file contains a particular type of data.**
   
   An entropy scan is useful for discovering compressed or encrypted data, but what about other data types such as machine code, symbol tables, sections of hardcoded ASCII strings, etc? Centrifuge takes advantage of the fact that in binary files, information encoded in a particular way is stored contiguously and uses scikit-learn's implementation of DBSCAN to locate these regions.
 - **Analyzing files with no metadata such as magic numbers, headers or other format information.**
  
   This includes most firmware, as well as corrupt files. Centrifuge does not depend on metadata or signatures of any kind.
 - **Investigating differences between different types of data using statistical methods or machine learning, or building a model or "profile" of a specific data type.**
  
   Does machine code differ in a systematic way from other types of information encoded in binary files? Can compressed data be distinguished from encrypted data? These questions can be investiged in an empirical way using Centrifuge.
 - **Visualizing file contents using Python plotting libraries such as Seaborn, Matplotlib and Altair**
  
   Rather than generate elaborate 2D or 3D visual representations of file contents using space-filling curves or cylindrical coordinate systems, Centrifuge creates data frames that contain the feature measurements of each cluster. The information contained in these data frames can be easily visualized with boxplots, violin plots, pairplots, histograms, density plots, scatterplots, barplots, cumulative distribution function (CDF) plots, etc.
 
 
