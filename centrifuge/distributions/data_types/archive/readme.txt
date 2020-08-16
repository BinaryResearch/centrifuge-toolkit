Examples of data types are machine code, ASCII text, compressed data, encrypted data, ELF debug info, etc.
After DBSCAN performs clustering, it will compare the byte distribution of each cluster to a data type reference.
This way no non-machine code information is compared with the reference distributions of the CPU architectures.
In other words, only if the cluster is identified to be data type "code" will an attempt to identify the
architecture be made.



To create the machine code reference, take a sample of size N bytes from all the architectures, merge them, then
build.


