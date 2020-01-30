/preprocess
===========
Data loading tools for network training and preprocessing.

# XMLManager

XMLManager supports iterative XML parsing, compatible with
given database's XML format.

XMLManager does XML reading, XML iterative parsing, dump into
smaller document.

# DataLoader

DataLoader read and iterate information from `DATA`, not `XML`.

# Data preprocessing process

1. Read data from XML by `XMLManager` and dump as file
2. Use `DataLoader` to read data from dumped files.


# Supporting Datasets

- KIBA
- DAVIS
- IBM