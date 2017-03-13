#!/bin/bash

#SBATCH -o log/%j.out
#SBATCH -e log/%j.out
#SBATCH -p titanx-long
#SBATCH -N 1
#SBATCH -c 18
#SBATCH -n 1
#SBATCH --gres=gpu:0

# Environment variables in use:
#
# DATASET
# NUM_CORES

CLUSTERING_ROOT=/home/hsu/Workspace1/clustering/data
CLUSTERING_DIR=$CLUSTERING_ROOT/$DATASET

if [ -z "$NUM_CORES" ]; then
    NUM_CORES=18
fi

ks=(10 50 100 200 500 1000 704 235)

methods_k_f=(AgglomerativeClustering Birch MiniBatchKMeans SpectralClustering)
methods_k_d=(SpectralClustering)
methods_nok_d=(AffinityPropagation DBSCAN)
methods_nok_f=(AffinityPropagation DBSCAN MeanShift)

FEAT_PATH=$CLUSTERING_DIR/X.mat
DIST_PATH=$CLUSTERING_DIR/dt.mat
NEG_C_PATH=$CLUSTERING_DIR/C.mat

for method in "${methods_k_f[@]}";
do
    for num_clusters in ${ks[@]}; 
    do 
        OUTPUT_PATH=$CLUSTERING_DIR/$method-0/$num_clusters.mat
        srun python clustering.py -o "$OUTPUT_PATH" -f "$FEAT_PATH" --nc $num_clusters -j $NUM_CORES -m $method
    done 
done 

for method in "${methods_k_d[@]}";
do
    for num_clusters in ${ks[@]}; 
    do 
        OUTPUT_PATH=$CLUSTERING_DIR/$method-1/$num_clusters.mat
        srun python clustering.py -o "$OUTPUT_PATH" -d "$DIST_PATH" --do-not-link "$NEG_C_PATH" --nc $num_clusters -j $NUM_CORES -m $method
    done 
done 

for method in "${methods_nok_d[@]}";
do
    OUTPUT_PATH=$CLUSTERING_DIR/$method-1/nok.mat
    srun python clustering.py -o "$OUTPUT_PATH" -d "$DIST_PATH" --do-not-link "$NEG_C_PATH" -j $NUM_CORES -m $method
done 

for method in "${methods_nok_f[@]}";
do
    OUTPUT_PATH=$CLUSTERING_DIR/$method-0/nok.mat
    srun python clustering.py -o "$OUTPUT_PATH" -f "$FEAT_PATH" -j $NUM_CORES -m $method
done 

