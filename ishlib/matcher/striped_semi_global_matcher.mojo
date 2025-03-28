"""Smith-Waterman local alignment."""
from sys.info import simdwidthof

from ishlib.matcher import Matcher, MatchResult
from ishlib.matcher.alignment import create_reversed
from ishlib.matcher.alignment.semi_global_aln.striped import (
    Profile,
    ScoringMatrix,
    ScoreSize,
    semi_global_aln_start_end,
)


# TODO: why is this a bit slower than the bench aligner, where the local on the same sequence, is much faster?


@value
struct StripedSemiGlobalMatcher[mut: Bool, //, origin: Origin[mut]](Matcher):
    alias SIMD_U8_WIDTH = simdwidthof[
        UInt8
    ]() // 4  # TODO: needs tuning on wider machines
    alias SIMD_U16_WIDTH = simdwidthof[
        UInt16
    ]() // 4  # TODO: needs tuning on wider machines
    var pattern: Span[UInt8, origin]
    var rev_pattern: List[UInt8]
    var rev_haystack_buffer: List[UInt8]
    var profile: Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH]
    var reverse_profile: Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH]
    var matrix: ScoringMatrix

    fn __init__(out self, pattern: Span[UInt8, origin]):
        var matrix = ScoringMatrix.all_ascii_default_matrix()
        self.matrix = matrix
        self.pattern = pattern
        self.rev_pattern = create_reversed(pattern)
        self.rev_haystack_buffer = List[UInt8]()
        var profile = Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH](
            self.pattern, self.matrix, ScoreSize.Adaptive
        )
        self.profile = profile
        var reverse_profile = Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH](
            self.rev_pattern, self.matrix, ScoreSize.Adaptive
        )
        self.reverse_profile = reverse_profile

    fn first_match(
        mut self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""

        self.rev_haystack_buffer.clear()
        self.rev_haystack_buffer.extend(haystack)
        self.rev_haystack_buffer.reverse()

        var result = semi_global_aln_start_end[do_saturation_check=False](
            reference=haystack,
            rev_reference=Span(self.rev_haystack_buffer),
            query_len=len(self.pattern),
            gap_open_penalty=3,
            gap_extension_penalty=1,
            profile=self.profile.profile_large.value(),
            rev_profile=self.reverse_profile.profile_large.value(),
            bias=self.profile.bias.cast[DType.uint16](),
            max_score=self.profile.max_score,
            min_score=self.profile.min_score,
            free_query_start_gaps=True,
            free_query_end_gaps=True,
            free_target_start_gaps=True,
            free_target_end_gaps=True,
            score_cutoff=Int32(len(self.pattern)),
        )
        if result.score >= len(self.pattern):
            return MatchResult(
                Int(result.target_start), Int(result.target_end + 1)
            )

        return None
