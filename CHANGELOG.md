
# Next

# v1.2.1 (06-25-2025)

- Fixed bug in logging module related to update from Mojo 24.3 to 24.4 replacing `write_args`.

# v1.2.0 (06-19-2025)

- Upgrade to Mojo 25.4
- Fix issue with [AVX512](https://github.com/BioRadOpenSource/ish/issues/50) via [PR](https://github.com/BioRadOpenSource/ish/pull/51) 
- [Apply](https://github.com/BioRadOpenSource/ish/pull/48) suggestion from @soraros to use llvm intrinsic for saturating add and subtract

# v1.1.1 (06-07-2025)

- Added a compile time `ISH_SIMD_TARGET` flag that can be set to `baseline` to
  force `ish` to use SSE sized widths.

# v1.1.0 (06-07-2025)

- Initial public release
- See https://github.com/BioRadOpenSource/ish/compare/v1.0.0...v1.1.0