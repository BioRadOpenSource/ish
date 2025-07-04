<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <img src="ish_logo.png" alt="Logo" width="250" height="250">

  <p align="center">
    Fast record filtering by alignment-scores 🔥
    <br/>

   ![Written in Mojo][language-shield]
   ![License Badge][license-shield]
   ![Build status][build-shield]
   ![CodeQL][codeql]
   <br/>
   ![Contributors Welcome][contributors-shield]

  </p>
</div>


`ish` is a CLI tool for searching for matches against records using different alignment methods.

## Build 

1. Install [pixi](https://pixi.sh/latest/installation/)

1. `pixi run build`
1. `./ish --help`

### Pixi / Conda install

```
pixi global install -c conda-forge -c https://repo.prefix.dev/modular-community -c https://conda.modular.com/max ish
# Or
conda install -c conda-forge -c https://repo.prefix.dev/modular-community -c https://conda.modular.com/max ish
```

For best performance it's recommended to build from source. 

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
		actgn: encodes searched inputs as nucleotides, matches are 2, mismatch is -2, Ns match anything.
		actgn0: encodes searched inputs as nucleotides, matches are 2, mismatch is -2, Ns don't count toward score.


	--score <Float> [Default: 0.8]
		The min score needed to return a match. Results >= this value will be returned. The score is the found alignment score / the optimal score for the given scoring matrix and gap-open / gap-extend penalty.

	--gap-open <Int> [Default: 3]
		Score penalty for opening a gap.

	--gap-extend <Int> [Default: 1]
		Score penalty for extending a gap.

	--match-algo <String> [Default: striped-semi-global]
		The algorithm to use for matching: [striped-local, striped-semi-global]

	--record-type <String> [Default: line]
		The input record type: [line, fastx]

	--threads <Int> [Default: 10]
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

- `striped-semi-global`: Striped Semi-global, SIMD accelerated, GPU accelerated when available, supports affine gaps and scoring matrices. Specify ends-free with the `--sg-ends-free` options.
- `striped-local`: Striped Smith-Waterman, SIMD accelerated, supports affine gaps and scoring matrices.

## Record Types

- `line`: match against one line at a time, a-la `grep`
- `fastx`: match against the sequence portion of FASTA or FASTQ records.


## ish-aligner

This is a benchmarking tool based on `parasail_aligner`.

> ⚠️ **Warning**
> 
> `ish-aligner` and all variations of it are for development purposes only.

## Further Reading

The associated paper can be found [here](https://www.biorxiv.org/content/10.1101/2025.06.04.657890v1).

## Future Work

- Support multiple queries
- Choose a better default between cpu and gpu / think about more. GPU crushes on big files / long running / many files, cpu is faster for small jobs
- Add ability to not skip dotfiles

## Rattler Build

For testing the build process for modular-community

```bash
pixi global install rattler-build
rattler-build build -c https://repo.prefix.dev/modular-community -c https://conda.modular.com/max -c conda-forge --skip-existing=all -r ./recipe.yaml
```

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[build-shield]: https://img.shields.io/circleci/build/github/BioRadOpenSource/ish 
[language-shield]: https://img.shields.io/badge/language-mojo-orange
[license-shield]: https://badgen.net/static/license/Apache-2.0/blue
[contributors-shield]: https://img.shields.io/badge/contributors-welcome!-blue
[codeql]: https://github.com/BioRadOpenSource/ish/workflows/CodeQL/badge.svg

