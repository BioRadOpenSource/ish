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
    var rev_pattern: List[UInt8]
    var profile: Profile
    var reverse_profile: Profile
    var matrix: ScoringMatrix

    fn __init__(out self, pattern: Span[UInt8, origin]):
        var matrix = ScoringMatrix.default_matrix(256, matched=2, mismatched=-2)
        self.matrix = matrix
        self.pattern = pattern
        self.rev_pattern = List[UInt8](capacity=len(pattern))
        for char in reversed(pattern):
            self.rev_pattern.append(char[])
        var profile = Profile(self.pattern, self.matrix, ScoreSize.Adaptive)
        self.profile = profile
        var reverse_profile = Profile(
            self.rev_pattern, self.matrix, ScoreSize.Adaptive
        )
        self.reverse_profile = reverse_profile

    fn first_match(
        mut self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""
        var result = ssw_align(
            self.profile,
            self.matrix,
            haystack,
            self.pattern,
            reverse_profile=self.reverse_profile,
        )
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
