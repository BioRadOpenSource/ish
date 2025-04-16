from gpu.host import DeviceContext

from ishlib.matcher import Matcher, MatchResult
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix, MatrixKind
from ishlib.matcher.alignment.global_aln.basic import (
    needleman_wunsch_parasail,
    needleman_wunsch_full_naive,
)
from ishlib.vendor.log import Logger


@value
struct BasicGlobalMatcher(Matcher):
    var pattern: List[UInt8]
    var scoring_matrix: ScoringMatrix

    fn __init__(
        out self,
        pattern: Span[UInt8],
        matrix_kind: MatrixKind = MatrixKind.ASCII,
    ):
        Logger.info("Performing matching with BasicGlobalMatcher.")
        self.scoring_matrix = matrix_kind.matrix()
        self.pattern = self.scoring_matrix.convert_ascii_to_encoding(pattern)

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
        if result.score >= len(pattern):
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
