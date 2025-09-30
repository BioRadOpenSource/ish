# Benchmarking

`pixi run bench-all` will download all needed data, and compile parasail.

## Prereqs and data

1. You will need to clone and build parasail

```bash
sudo apt-get install libtool
git clone https://github.com/jeffdaily/parasail
cd parasail
autoreconf -fi
./configure
make -j $(nproc)
```

2. You will need to have mojo and pixi installed. See the Modular website for install instructions.

## Data

1. Create the benchmarking and data dir

```bash 
mkdir -p bench/data && cd bench/data
```

2. The sequences used for benchmarking are the same as those used in the [parasail paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-0930-z). Query sequences perform the following: 

```bash
git clone https://github.com/jeffdaily/parasail
```

The sequences are will be in `bench/data/parasail/data`.

3. The reference data can be gotten by:

```bash
mkdir refdata && cd refdata
curl https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2015_11/knowledgebase/uniprot_sprot-only2015_11.tar.gz --output uniprot_sprot-only2015_11.tar.gz
tar -xvzf uniprot_sprot-only2015_11.tar.gz
```
