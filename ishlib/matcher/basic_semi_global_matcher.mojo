from gpu.host import DeviceContext

from ishlib.matcher import Matcher, MatchResult
from ishlib.matcher.alignment import create_reversed
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix, MatrixKind
from ishlib.matcher.alignment.semi_global_aln.basic import (
    semi_global_parasail_start_end,
)
from ishlib.vendor.log import Logger


@value
struct BasicSemiGlobalMatcher(Matcher):
    var pattern: List[UInt8]
    var rev_pattern: List[UInt8]
    var max_score: Int
    var scoring_matrix: ScoringMatrix
    var gap_open: UInt
    var gap_extend: UInt
    var _score_threshold: Float32

    fn __init__(
        out self,
        pattern: Span[UInt8],
        score_threshold: Float32,
        matrix_kind: MatrixKind = MatrixKind.ASCII,
        gap_open: UInt = 3,
        gap_extend: UInt = 1,
    ):
        Logger.info("Performing matching with BasicSemiGlobalMatcher.")
        self._score_threshold = score_threshold
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.scoring_matrix = matrix_kind.matrix()
        (
            self.pattern,
            self.max_score,
        ) = self.scoring_matrix.convert_ascii_to_encoding_and_score(pattern)
        self.rev_pattern = create_reversed(self.pattern)

    fn first_match(
        read self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""

        var rev_haystack = create_reversed(haystack)

        var result = semi_global_parasail_start_end[DType.int16](
            self.pattern,
            haystack,
            self.rev_pattern,
            rev_haystack,
            self.scoring_matrix,
            gap_open_penalty=-Int(self.gap_open),
            gap_extension_penalty=-Int(self.gap_extend),
            # TODO: How should these be set on the CLI?
            # Rust bio-aligner has a nice concept around clipping that might make more sense here.
            free_query_start_gaps=True,
            free_query_end_gaps=True,
            free_target_start_gaps=True,
            free_target_end_gaps=True,
            score_cutoff=len(pattern),
        )
        # TODO: Update this
        if (
            result
            and Float32(result.value().score) / self.max_alignment_score()
            >= self.score_threshold()
        ):
            return MatchResult(
                result.value().coords.value().start,
                result.value().coords.value().end,
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
