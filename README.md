# ish 🔥

Accelerated alignment on the CLI.

:::warning
`ish` is under active development.
::::


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

## Match Methods

- `ssw`: Striped Smith Waterman, SIMD accelerated, supports affine gaps and scoring matrices. TODO: non-ascii scoring matrix
- `sw_local`: Classic full matrix dynamic programming Smith-Waterman alignment, does not support affine gaps.
- `naive_exact`: 

## Record Types

- `line`: match against one line at a time, a-la `grep`
- `fasta`: match against the sequence portion of fastq records.


## ish-aligner

This is a benchmarking tool based on `parasail_aligner`.

:::Work in progress
:::

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
