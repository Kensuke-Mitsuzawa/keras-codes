#!/usr/bin/env bash
wget http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/data/20170201.tar.bz2 -O latest-wikipedia-entity-vector.tar.bz2
bzip2 -dc latest-wikipedia-entity-vector.tar.bz2| tar xvf -