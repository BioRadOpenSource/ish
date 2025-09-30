"""Smith-Waterman local alignment."""
from gpu.host import DeviceContext

from ishlib.vendor.log import Logger
from ishlib.matcher import Matcher, MatchResult
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix, MatrixKind
from ishlib.matcher.alignment.local_aln.basic import smith_waterman


@fieldwise_init
struct BasicLocalMatcher(Matcher):
    var pattern: List[UInt8]
    var max_score: Int
    var _scoring_matrix: ScoringMatrix
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
        Logger.info("Perorming matching with BasicLocalMatcher.")
        Logger.warn(
            "Basic local matcher does not use a scoring matrix and defaults to"
            " match 2, mismatch -3, gap -3"
        )
        self._score_threshold = score_threshold
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self._scoring_matrix = matrix_kind.matrix()
        var encoding_and_score = (
            self._scoring_matrix.convert_ascii_to_encoding_and_score(pattern)
        )
        self.pattern = encoding_and_score[0].take()
        self.max_score = encoding_and_score[1].take()

    fn first_match(
        read self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""
        var result = smith_waterman(
            haystack,
            pattern,
            match_score=2,
            mismatch_score=-2,
            gap_penalty=-Int(self.gap_open),
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
        return self._scoring_matrix.convert_ascii_to_encoding(value)

    @always_inline
    fn convert_encoding_to_ascii(read self, value: UInt8) -> UInt8:
        """Convert an encoded byte to an ascii byte."""
        return self._scoring_matrix.convert_encoding_to_ascii(value)

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
