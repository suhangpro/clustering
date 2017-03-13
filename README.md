# clustering 
```
usage: clustering.py [-h] -o OUTPUT [-f F] [-d D] [--do-not-link DO_NOT_LINK]
                     [--must-link MUST_LINK] [-z] [-v VERBOSE] [-j N_JOBS]
                     [--metric METRIC] [--nc NC] [--max-iter MAX_ITER]
                     [--linkage LINKAGE] [-u]
                     [-m {AffinityPropagation,AgglomerativeClustering,Birch,DBSCAN,KMeans,MiniBatchKMeans,MeanShift,SpectralClustering}]

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output path
  -f F, --feature F     path to feature matrix
  -d D, --distance D    path to distance matrix, only compatible with methods:
                        AffinityPropagation/DBSCAN/SpectralClustering
  --do-not-link DO_NOT_LINK
                        path to do-not-link constraint matrix, only works with
                        methods: AffinityPropagation/DBSCAN/SpectralClustering
  --must-link MUST_LINK
                        path to must-link constraint matrix, only works with
                        methods: AffinityPropagation/DBSCAN/SpectralClustering
  -z, --zero-based      output with zero-based indexing if enabled
  -v VERBOSE, --verbose VERBOSE
                        verbose level 0|1[default]|2
  -j N_JOBS, --n-jobs N_JOBS
                        number of parallel jobs to run (default: -1)
  --metric METRIC       metric for computing distances (default: euclidean)
  --nc NC, --n-clusters NC
                        number of clusters, required for AgglomerativeClusteri
                        ng/Birch/KMeans/MiniBatchKMeans/SpectralClustering
  --max-iter MAX_ITER   maximum number of iterations (default: KMeans:300/Mini
                        BatchKMeans:100/AffinityPropagation:200)
  --linkage LINKAGE     which linkage criterion to use (default: ward)
  -u, --update          unless enabled, skip if output already exists
  -m {AffinityPropagation,AgglomerativeClustering,Birch,DBSCAN,KMeans,MiniBatchKMeans,MeanShift,SpectralClustering}, --method {AffinityPropagation,AgglomerativeClustering,Birch,DBSCAN,KMeans,MiniBatchKMeans,MeanShift,SpectralClustering}
                        specifies which clustering algorithm to use (default:
                        KMeans)
```

# clustering (legacy version)
```
usage: clustering_legacy.py [-h] [-c CONSTRAINTS] [-z] [-d] [-n] [-v]
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

# connectivity
```
usage: connectivity.py [-h] [-v] [-d DELIMITER]
                       [-s {pre_shuffle,trial_and_error}]
                       nodes runs

Graph connectivity simulation.

positional arguments:
  nodes                 Number of nodes in the graph
  runs                  Number of runs to simulate

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Output runtime information when True
  -d DELIMITER, --delimiter DELIMITER
                        Delimiter in output
  -s {pre_shuffle,trial_and_error}, --sample-scheme {pre_shuffle,trial_and_error}
                        Sampling scheme [default: 'trial_and_error']
```

# feature ranking with mutual information
```
usage: rank_mi.py [-h] [-f FEATURE] [--mi MI] [-o OUTPUT] [-k TOP_K]
                  [-d DELIMITER] [-q NUM_BINS] [-n NUM_SAMPLES] [-z] [-v]
                  [-p {none,abs,square}]
                  [-t { mutual_info_score,adjusted_mutual_info_score,normalized_mutual_info_score}]
                  [-s {random,smallest}] [--use-percentile]

optional arguments:
  -h, --help            show this help message and exit
  -f FEATURE, --feature FEATURE
                        path to feature matrix
  --mi MI, --mutual-information MI
                        output path for mutual information if specified
  -o OUTPUT, --output OUTPUT
                        top reranked features greedily selected wrt mutual
                        information [default: '-' (stdout)]
  -k TOP_K, --top-k TOP_K
                        top k features [default: -1 (all)]
  -d DELIMITER, --delimiter DELIMITER
                        only used when output to stdout [default: ',']
  -q NUM_BINS, --num-bins NUM_BINS
                        quantization levels [default: 20]
  -n NUM_SAMPLES, --num-samples NUM_SAMPLES
                        number of samples to use [default: -1 (all)]
  -z, --zero-based      output with zero-based indexing if enabled
  -v, --verbose         verbose
  -p {none,abs,square}, --pre-process {none,abs,square}
                        feature pre-processing [default: 'none']
  -t { mutual_info_score,adjusted_mutual_info_score,normalized_mutual_info_score}, --score-type { mutual_info_score,adjusted_mutual_info_score,normalized_mutual_info_score}
                        mutual information type [default: 'standard']
  -s {random,smallest}, --start-with {random,smallest}
                        starting strategy [default: 'smallest']
  --use-percentile      discretize using percentile if specified
```

