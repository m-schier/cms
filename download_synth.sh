#!/bin/sh

wget http://cs.joensuu.fi/sipu/datasets/Aggregation.txt -P data
wget http://cs.joensuu.fi/sipu/datasets/jain.txt -P data
wget http://cs.joensuu.fi/sipu/datasets/s4.txt -P data
wget http://cs.joensuu.fi/sipu/datasets/s-originals.zip -P data

unzip data/s-originals.zip -d data
