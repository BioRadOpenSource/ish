"""Smith-Waterman local alignment."""
from gpu.host import DeviceContext
from sys.info import (
    simdwidthof,
)
from utils import StringSlice

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
    ]() // 1  # TODO: needs tuning based on query length
    alias SIMD_U16_WIDTH = simdwidthof[UInt16]() // 1

    var pattern: List[UInt8]
    var rev_pattern: List[UInt8]
    var max_score: Int
    var profile: Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH]
    var reverse_profile: Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH]
    var matrix: ScoringMatrix
    var gap_open: UInt
    var gap_extend: UInt
    var _score_threshold: Float32

    fn __init__(
        out self,
        pattern: Span[UInt8, origin],
        score_threshold: Float32,
        matrix_kind: MatrixKind = MatrixKind.ASCII,
        gap_open: UInt = 3,
        gap_extend: UInt = 1,
    ):
        Logger.info("Performing matching with StripedLocalMatcher")
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self._score_threshold = score_threshold
        self.matrix = matrix_kind.matrix()
        (
            self.pattern,
            self.max_score,
        ) = self.matrix.convert_ascii_to_encoding_and_score(pattern)
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
            score_cutoff=-1,
        )

        if (
            result
            and (
                result.value().ref_begin1 >= 0 and result.value().ref_end1 >= 0
            )
            and Float32(result.value().score1) / self.max_alignment_score()
            >= self.score_threshold()
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

    @always_inline
    fn max_alignment_score(read self) -> Int:
        return self.max_score

    @always_inline
    fn score_threshold(read self) -> Float32:
        """Returns the score threshold needed to be concidered a match."""
        return self._score_threshold
