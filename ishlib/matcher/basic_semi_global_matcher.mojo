"""Needleman-Wunsch global alignment."""
from ishlib.matcher import Matcher, MatchResult
from ishlib.matcher.alignment import create_reversed
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix
from ishlib.matcher.alignment.semi_global_aln.basic import (
    semi_global_parasail_start_end_end,
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
        mut self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""

        var rev_haystack = create_reversed(haystack)

        var result = semi_global_parasail_start_end_end[DType.int16](
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
        )
        if result.score >= len(pattern):
            return MatchResult(
                result.coords.value().start, result.coords.value().end
            )

        return None
