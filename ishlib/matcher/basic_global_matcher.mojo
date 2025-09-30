from gpu.host import DeviceContext

from ishlib.matcher import Matcher, MatchResult
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix, MatrixKind
from ishlib.matcher.alignment.global_aln.basic import (
    needleman_wunsch_parasail,
    needleman_wunsch_full_naive,
)
from ishlib.vendor.log import Logger


@fieldwise_init
struct BasicGlobalMatcher(Matcher):
    var pattern: List[UInt8]
    var scoring_matrix: ScoringMatrix
    var max_score: Int
    var _score_threshold: Float32

    fn __init__(
        out self,
        pattern: Span[UInt8],
        score_threshold: Float32,
        matrix_kind: MatrixKind = MatrixKind.ASCII,
    ):
        Logger.info("Performing matching with BasicGlobalMatcher.")
        self._score_threshold = score_threshold
        self.scoring_matrix = matrix_kind.matrix()

        var encoding_and_score = (
            self.scoring_matrix.convert_ascii_to_encoding_and_score(pattern)
        )
        self.pattern = encoding_and_score[0].take()
        self.max_score = encoding_and_score[1].take()

    fn first_match(
        read self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""
        var result = needleman_wunsch_parasail[DType.int16](
            self.pattern,
            haystack,
            self.scoring_matrix,
            gap_open_penalty=-3,
            gap_extension_penalty=-1,
        )
        if (
            Float32(result.score) / Float32(self.max_alignment_score())
            >= self.score_threshold()
        ):
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
