from gpu.host import DeviceContext

from ishlib.matcher import Matcher, MatchResult
from ishlib.matcher.alignment import create_reversed
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix
from ishlib.matcher.alignment.semi_global_aln.basic import (
    semi_global_parasail_start_end,
)


@value
struct BasicSemiGlobalMatcher(Matcher):
    var pattern: List[UInt8]
    var rev_pattern: List[UInt8]
    var scoring_matrix: ScoringMatrix

    fn __init__(out self, pattern: Span[UInt8]):
        self.pattern = List(pattern)  # assuming ascii matrix
        self.rev_pattern = create_reversed(pattern)
        self.scoring_matrix = ScoringMatrix.all_ascii_default_matrix()

    fn first_match(
        read self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""

        var rev_haystack = create_reversed(haystack)

        var result = semi_global_parasail_start_end[DType.int16](
            self.pattern,
            haystack,  # assuming ascii matrix so we can skip encoding
            self.rev_pattern,
            rev_haystack,
            self.scoring_matrix,
            gap_open_penalty=-3,
            gap_extension_penalty=-1,
            # TODO: How should these be set on the CLI?
            # Rust bio-aligner has a nice concept around clipping that might make more sense here.
            free_query_start_gaps=True,
            free_query_end_gaps=True,
            free_target_start_gaps=True,
            free_target_end_gaps=True,
            score_cutoff=len(pattern),
        )
        if result and result.value().score >= len(pattern):
            print(
                result.value().score, "none", result.value().coords.value().end
            )
            return MatchResult(
                result.value().coords.value().start,
                result.value().coords.value().end,
            )

        return None

    @always_inline
    fn convert_ascii_to_encoding(read self, value: UInt8) -> UInt8:
        """Convert an ascii byte to an encoded byte."""
        return self.scoring_matrix.convert_ascii_to_encoding(value)

    @always_inline
    fn convert_encoding_to_ascii(read self, value: UInt8) -> UInt8:
        """Convert an encoded byte to an ascii byte."""
        return self.scoring_matrix.convert_encoding_to_ascii(value)
