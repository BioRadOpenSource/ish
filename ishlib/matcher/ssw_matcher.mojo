"""Smith-Waterman local alignment."""
from ishlib.matcher import Matcher, MatchResult
from ishlib.matcher.alignment.ssw_align import (
    ssw_align,
    Profile,
    ScoringMatrix,
    ScoreSize,
)


@value
struct SSWMatcher[mut: Bool, //, origin: Origin[mut]](Matcher):
    var pattern: Span[UInt8, origin]
    var profile: Profile[origin]
    var matrix: ScoringMatrix

    fn __init__(out self, pattern: Span[UInt8, origin]):
        var matrix = ScoringMatrix.default_matrix(256, matched=2, mismatched=-2)
        self.matrix = matrix
        self.pattern = pattern
        var profile = Profile[origin](
            self.pattern, self.matrix, ScoreSize.Adaptive
        )
        self.profile = profile

    fn first_match(
        mut self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""
        var result = ssw_align(self.profile, self.matrix, haystack)
        if (
            result
            and (
                result.value().ref_begin1 >= 0 and result.value().ref_end1 >= 0
            )
            and result.value().score1 >= len(self.pattern)
            # and result.value().score1 > result.value().score2
        ):
            return MatchResult(
                Int(result.value().ref_begin1), Int(result.value().ref_end1 + 1)
            )

        return None
