
Next Steps:
- Pull the C code and set up a little mini example that can dump the intermediate steps
- Then compare against the steps in Mojo
- Fixup Mojo
- Prepare some visualizations about how on earth this works.


Idea attribution:
- SSW lib
- BWA-Mem - stores the query vecs on the profile as well
other speedups and attributions?

Novel things:
- AVX512
- precompute the reverse profile for finding starts
    - is this actually good? or only for my test data?

Could try:
- SIMDify the secondary score lookup