# clustering
```
usage: clustering.py [-h] [-c CONSTRAINTS] [-z] [-d] [-n] [-v]
                     similarity output threshold

Constraint Similarity-Based Clustering.

positional arguments:
  similarity            Path to a file containing similarity matrix
  output                Output file path
  threshold             Threshold, or the number of clusters when -n is turned
                        on

optional arguments:
  -h, --help            show this help message and exit
  -c CONSTRAINTS, --constraints CONSTRAINTS
                        Path to a file containing constraints
  -z, --zero-based      Cluster ID will start from 0 if turned on
  -d, --distance        Distance is given rather than similarity
  -n, --num-clusters    Number of clusters should be given if True
  -v, --verbose         Output runtime information when True
  ```
