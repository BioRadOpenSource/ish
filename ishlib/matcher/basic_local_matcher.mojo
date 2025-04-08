"""Smith-Waterman local alignment."""
from gpu.host import DeviceContext

from ishlib.matcher import Matcher, MatchResult
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix
from ishlib.matcher.alignment.local_aln.basic import smith_waterman


@value
struct BasicLocalMatcher(Matcher):
    var pattern: List[UInt8]
    var scoring_matrix: ScoringMatrix

    fn __init__(out self, pattern: Span[UInt8]):
        self.pattern = List(pattern)  # assuming ascii matrix
        self.scoring_matrix = ScoringMatrix.all_ascii_default_matrix()

    fn first_match(
        read self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""
        var result = smith_waterman(
            haystack, pattern, match_score=2, mismatch_score=-2, gap_penalty=-3
        )
        if result.score == len(pattern):
            return MatchResult(
                result.coords.value().start, result.coords.value().end
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
