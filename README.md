# ish 🔥

Accelerated alignment on the CLI.

> ⚠️ **Warning**
> 
> `ish` is under active development.


`ish` is a CLI tool for searching for matches against records using different alignment methods.

## Install

1. Install the mojo build tool [magic](https://docs.modular.com/magic/)

1. `magic run build`
1. `./ish --help`

### Conda install

Mojo packages are hosted in a conda repo, once this tool is baked it can be published to the Modular channel and then installed via conda anywhere.

## Usage

```sh
❯ ./ish --help
ish
Search for inexact patterns in files.

ARGS:
        <ARGS (>=1)>...
                Pattern to search for, then any number of files or directories to search.
FLAGS:
        --help <Bool> [Default: False]
                Show help message

        --verbose <Bool> [Default: False]
                Verbose logging output.

OPTIONS:
        --scoring-matrix <String> [Default: ascii]
                The scoring matrix to use.
                ascii: does no encoding of input bytes, matches are 2, mismatch is -2.
                blosum62: encodes searched inputs as amino acids and uses the classic Blosum62 scoring matrix.
                actgn: encodes searched inputs as nucleotides, matches are 2, mismatch is -2, Ns match anything


        --score <Float> [Default: 0.8]
                The min score needed to return a match. Results >= this value will be returned. The score is the found alignment score / the optimal score for the given scoring matrix and gap-open / gap-extend penalty.

        --gap-open <Int> [Default: 3]
                Score penalty for opening a gap.

        --gap-extend <Int> [Default: 1]
                Score penalty for extending a gap.

        --match-algo <String> [Default: striped-semi-global]
                The algorithm to use for matching: [naive_exact, striped-local, basic-local, basic-global, basic-semi-global, striped-semi-global]

        --record-type <String> [Default: line]
                The input record type: [line, fasta]

        --threads <Int> [Default: 24]
                The number of threads to use. Defaults to the number of physical cores.

        --batch-size <Int> [Default: 268435456]
                The number of bytes in a parallel processing batch. Note that this may use 2-3x this amount to account for intermediate transfer buffers.

        --max-gpus <Int> [Default: 0]
                The max number of GPUs to try to use. If set to 0 this will ignore any found GPUs. In general, if you have only one query then there won't be much using more than 1 GPU. GPUs won't always be faster than CPU parallelization depending on the profile of data you are working with.

        --output-file <String> [Default: /dev/stdout]
                The file to write the output to, defaults to stdout.

        --sg-ends-free <String> [Default: FFTT]
                The ends-free for semi-global alignment, if used. The free ends are: (query_start, query_end, target_start, target_end). These must be specified with a T or F, all four must be specified. By default this target ends are free.
```

```sh
# Some actual usage.
❯ ./ish blosum62 ./ish_bench_aligner.mojo 
./ish_bench_aligner.mojo:94             default_value=String("Blosum50"),
./ish_bench_aligner.mojo:96                 "Scoring matrix to use. Currently supports: [Blosum50,"
./ish_bench_aligner.mojo:97                 " Blosum62, ACTGN]"
./ish_bench_aligner.mojo:379     if matrix_name == "Blosum50":
./ish_bench_aligner.mojo:380         matrix = ScoringMatrix.blosum50()
./ish_bench_aligner.mojo:381     elif matrix_name == "Blosum62":
./ish_bench_aligner.mojo:382         matrix = ScoringMatrix.blosum62()
./ish_bench_aligner.mojo:390     ## Assuming we are using Blosum50 AA matrix for everything below this for now.
```

> 🔥 **Note**
>
> The `filepath:linenumber` in the match allows you to `cmd-click` on the match and have vscode open the file at that location.

## Match Methods

- `striped-local`: Striped Smith-Waterman, SIMD accelerated, supports affine gaps and scoring matrices. TODO: non-ascii scoring matrix
- `basic-local`: Classic full matrix dynamic programming Smith-Waterman alignment, does not support affine gaps.
- `basic-global`: Classic Needleman-Wunsch global alignment.
- TODO: update

## Record Types

- `line`: match against one line at a time, a-la `grep`
- `fasta`: match against the sequence portion of fastq records.


## ish-aligner

This is a benchmarking tool based on `parasail_aligner`.

> ⚠️ **Warning**
> 
> `ish-aligner` is under active development.

## Future Work

Next Steps:
- Add alignment output - like a nice viz
- Add FASTQ support for record type
- Add SAM support for record type
- Add CSV support, matching against individual columns?
- Add more matchers
- Parallelize over files when gpus are available

Idea attribution:
- SSW lib / parasail
- BWA-Mem - stores the query vecs on the profile as well
other speedups and attributions?

Novel things:
- AVX512
- oversubscribed SSE2 to mimic AVX2
- precompute the reverse profile for finding starts
    - is this actually good? or only for my test data?
- Dynamic selection of the simd width based on the query length
- the ish tool itself, doing index-free alignments

## TODO 

- FASTQ support
- Support muliple queries
- Choose a better default between cpu and gpu / think about more. GPU crushes on big files / long running / many files, cpu is faster for small jobs
- Add ability to not skip dotfiles


TODO to make promises in the paper:

- Read targets from stdin
- FASTQ support

Comparable tools:
- https://github.com/laurikari/tre/ (the version of agrep this compiles)
- https://github.com/fulcrumgenomics/fqgrep, no protein support, no fuzzy match
- https://github.com/Rbfinch/grepq, only fasta/fastq, no fuzzy match
