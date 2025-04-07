"""Smith-Waterman local alignment."""
from sys.info import simdwidthof

from ishlib.matcher import Matcher, MatchResult
from ishlib.matcher.alignment.local_aln.striped import (
    ssw_align,
    Profile,
    ScoringMatrix,
    ScoreSize,
)


@value
struct StripedLocalMatcher[mut: Bool, //, origin: Origin[mut]](Matcher):
    alias SIMD_U8_WIDTH = simdwidthof[
        UInt8
    ]() // 1  # TODO: needs tuning on wider machines (avx512) for example
    alias SIMD_U16_WIDTH = simdwidthof[UInt16]() // 1
    var pattern: Span[UInt8, origin]
    var rev_pattern: List[UInt8]
    var profile: Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH]
    var reverse_profile: Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH]
    var matrix: ScoringMatrix

    fn __init__(out self, pattern: Span[UInt8, origin]):
        var matrix = ScoringMatrix.all_ascii_default_matrix()
        self.matrix = matrix
        self.pattern = pattern
        self.rev_pattern = List[UInt8](capacity=len(pattern))
        for char in reversed(pattern):
            self.rev_pattern.append(char[])
        var profile = Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH](
            self.pattern, self.matrix, ScoreSize.Adaptive
        )
        self.profile = profile
        var reverse_profile = Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH](
            self.rev_pattern, self.matrix, ScoreSize.Adaptive
        )
        self.reverse_profile = reverse_profile

    fn first_match(
        read self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""
        var result = ssw_align(
            self.profile,
            self.matrix,
            haystack,
            self.pattern,
            reverse_profile=self.reverse_profile,
            score_cutoff=Int32(len(self.pattern)) - 1,
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

    @always_inline
    fn convert_ascii_to_encoding(read self, value: UInt8) -> UInt8:
        """Convert an ascii byte to an encoded byte."""
        return self.matrix.convert_ascii_to_encoding(value)

    @always_inline
    fn convert_encoding_to_ascii(read self, value: UInt8) -> UInt8:
        """Convert an encoded byte to an ascii byte."""
        return self.matrix.convert_encoding_to_ascii(value)
