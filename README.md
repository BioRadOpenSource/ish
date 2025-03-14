
Next Steps:
- Add FASTQ support for recrod type
- Add ability to set scoring matrix type (ascii, actgn, bl50, bl62)
- Prepare some visualizations about how on earth this works.
- Add tty detection?


- Work on the plots outlined in obsidian

Idea attribution:
- SSW lib / parasail
- BWA-Mem - stores the query vecs on the profile as well
other speedups and attributions?

Novel things:
- AVX512
- precompute the reverse profile for finding starts
    - is this actually good? or only for my test data?
- Dynamic selection of the simd width based on the query length
- the ish tool itself, doing index-free alignments

Could try:
- SIMDify the secondary score lookup