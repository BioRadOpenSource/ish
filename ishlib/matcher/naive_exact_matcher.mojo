"""Naive linear exact match search."""
from ishlib.matcher import Matcher, MatchResult
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix


@value
struct NaiveExactMatcher(Matcher):
    var pattern: List[UInt8]
    var scoring_matrix: ScoringMatrix

    fn __init__(out self, pattern: Span[UInt8]):
        self.pattern = List(pattern)  # assuming ascii matrix
        self.scoring_matrix = ScoringMatrix.all_ascii_default_matrix()

    fn first_match(
        read self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""

        for h in range(0, len(haystack)):
            var matched = True
            for p in range(0, len(pattern)):
                if pattern[p] != haystack[h + p]:
                    matched = False
                    break
            if matched:
                return MatchResult(h, h + len(pattern))

        return None

    @always_inline
    fn convert_ascii_to_encoding(read self, value: UInt8) -> UInt8:
        """Convert an ascii byte to an encoded byte."""
        return self.scoring_matrix.convert_ascii_to_encoding(value)

    @always_inline
    fn convert_encoding_to_ascii(read self, value: UInt8) -> UInt8:
        """Convert an encoded byte to an ascii byte."""
        return self.scoring_matrix.convert_encoding_to_ascii(value)
