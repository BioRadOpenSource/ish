"""Smith-Waterman local alignment."""
from gpu.host import DeviceContext
from sys.info import simdwidthof

from ishlib.vendor.log import Logger
from ishlib.matcher import Matcher, MatchResult
from ishlib.matcher.alignment import create_reversed
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix, MatrixKind
from ishlib.matcher.alignment.local_aln.striped import (
    ssw_align,
    Profile,
    ScoreSize,
)


@value
struct StripedLocalMatcher[mut: Bool, //, origin: Origin[mut]](Matcher):
    alias SIMD_U8_WIDTH = simdwidthof[
        UInt8
    ]() // 1  # TODO: needs tuning on wider machines (avx512) for example
    alias SIMD_U16_WIDTH = simdwidthof[UInt16]() // 1
    var pattern: List[UInt8]
    var rev_pattern: List[UInt8]
    var profile: Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH]
    var reverse_profile: Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH]
    var matrix: ScoringMatrix
    var gap_open: UInt
    var gap_extend: UInt

    fn __init__(
        out self,
        pattern: Span[UInt8, origin],
        matrix_kind: MatrixKind = MatrixKind.ASCII,
        gap_open: UInt = 3,
        gap_extend: UInt = 1,
    ):
        Logger.info("Performing matching with StripedLocalMatcher")
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.matrix = matrix_kind.matrix()
        self.pattern = self.matrix.convert_ascii_to_encoding(pattern)
        self.rev_pattern = create_reversed(self.pattern)
        var profile = Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH](
            self.pattern, self.matrix, ScoreSize.Adaptive
        )
        self.profile = profile
        var reverse_profile = Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH](
            self.rev_pattern, self.matrix, ScoreSize.Adaptive
        )
        self.reverse_profile = reverse_profile

    fn first_match(
        read self, haystack: Span[UInt8], _pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""
        var result = ssw_align(
            self.profile,
            self.matrix,
            haystack,
            self.pattern,
            gap_open_penalty=self.gap_open,
            gap_extension_penalty=self.gap_extend,
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

    @always_inline
    fn encoded_pattern(ref self) -> Span[UInt8, __origin_of(self)]:
        return Span[UInt8, __origin_of(self)](
            ptr=self.pattern.unsafe_ptr(), length=len(self.pattern)
        )
