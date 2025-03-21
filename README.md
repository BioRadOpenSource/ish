# ish 🔥

Accelerated alignment on the CLI.

> ⚠️ **Warning**
> 
> `ish` is under active development.


`ish` is a CLI tool for searching for matches against records using different alignment methods.

## Install

1. Install the mojo build tool [magic](https://docs.modular.com/magic/)
1. `ish` relies on a dev version of ExtraMojo, for now. This will be updated soon and just be a normal dep.

```
git clone git@github.com:ExtraMojo/ExtraMojo.git
cd ExtraMojo
git checkout feat/check_cli_arg_len
```
1. Update the paths to `ExtraMojo` in the `mojoproject.toml` file.
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
                Files to search for the given pattern.
FLAGS:
        --help <Bool> [Default: False]
                Show help message

OPTIONS:
        --pattern <String> [Required]
                The pattern to search for.

        --min-score <Int> [Default: 1]
                The min score needed to return a match.

        --match-algo <String> [Default: ssw]
                The algorithm to use for matching: [naive_exact, ssw, sw_local]

        --record-type <String> [Default: line]
                The input record type: [line, fasta]
```

```sh
# Some actual usage.
❯ ./ish --pattern "blosum62" ---match-algo ssw ./ish_bench_aligner.mojo 
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
- `naive_exact`: 

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
- Add ability to set score more easily as a fraction of length or some toher metrics
- Add alignment output - like a nice viz
- Add FASTQ support for record type
- Add SAM support for record type
- Add CSV support, matching against individual columns?
- Add more matchers
- Add ability to set scoring matrix type (ascii, actgn, bl50, bl62)
- Add tty detection
- Turn colorization on and off based on tty
- Switch to turn on/off the `filepath:linenumber` output

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

## TODO Tomorrow
- intrisics may have been working? add back in the assumes?
- add aarch64 linux support

```sh

```