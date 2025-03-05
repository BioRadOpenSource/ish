"""Smith-Waterman local alignment."""
from ishlib.matcher import Matcher, MatchResult
from ishlib.matcher.alignment.ssw_align import (
    ssw_align,
    Profile,
    ScoringMatrix,
    ScoreSize,
)

alias ASCII_MATRIX = ScoringMatrix.default_matrix(256, matched=2, mismatched=-2)


@value
struct SSWMatcher[mut: Bool, //, origin: Origin[mut]](Matcher):
    var pattern: Span[UInt8, origin]
    var profile: Profile[origin, StaticConstantOrigin]

    fn __init__(out self, pattern: Span[UInt8, origin]):
        self.pattern = pattern
        var profile = Profile[origin](
            self.pattern, ASCII_MATRIX, ScoreSize.Adaptive
        )
        self.profile = rebind[Profile[origin, StaticConstantOrigin]](profile)

    fn first_match(
        mut self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""
        var result = ssw_align[__origin_of(self.profile)](
            self.profile, haystack
        )
        if result:
            return MatchResult(
                Int(result.value().ref_begin1), Int(result.value().ref_end1 + 1)
            )
        # var result = smith_waterman(haystack, pattern)
        # if result.score == len(pattern):
        #     return MatchResult(
        #         result.coords.value()[0], result.coords.value()[1]
        #     )

        return None
