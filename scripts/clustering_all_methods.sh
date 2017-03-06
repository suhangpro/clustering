#!/bin/bash

# Environment variables in use:
#
# FEAT_PATH
# DIST_PATH
# DO_NOT_LINK_PATH
# OUTPUT_DIR
# NUM_CLUSTERS

if [ -z "$NUM_CLUSTERS" ]; then
    NUM_CLUSTERS=-1
fi

methods_available=(AffinityPropagation AgglomerativeClustering Birch DBSCAN KMeans MiniBatchKMeans MeanShift SpectralClustering)
methods_can_take_dist=(AffinityPropagation DBSCAN SpectralClustering)
methods_require_n_clusters=(AgglomerativeClustering Birch KMeans MiniBatchKMeans SpectralClustering)

for method in "${methods_available[@]}";
do
    if [ -n "$DIST_PATH" ] || [ -n "$DO_NOT_LINK_PATH" ]; then
        if ! echo ${methods_can_take_dist[@]}  | grep -q -w "$method"; then
            continue
        fi
    fi
    if [ "$NUM_CLUSTERS" -lt 1 ] ; then
        if echo ${methods_require_n_clusters[@]}  | grep -q -w "$method"; then
            continue
        fi
    fi
    python clustering.py -o "$OUTPUT_DIR" -f "$FEAT_PATH" -d "$DIST_PATH" --do-not-link "$DO_NOT_LINK_PATH" --nc $NUM_CLUSTERS -m $method
done
