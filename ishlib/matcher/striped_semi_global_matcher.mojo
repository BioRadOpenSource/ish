"""Smith-Waterman local alignment."""
from gpu.host import DeviceContext
from sys.info import simdwidthof

from ishlib.matcher import Matcher, MatchResult
from ishlib.matcher.alignment import create_reversed
from ishlib.gpu.kernels.semi_global import gpu_align_coarse
from ishlib.matcher.alignment.semi_global_aln.striped import (
    Profile,
    ScoringMatrix,
    ScoreSize,
    semi_global_aln_start_end,
    semi_global_aln,
)


# TODO: why is this a bit slower than the bench aligner, where the local on the same sequence, is much faster?


@value
struct StripedSemiGlobalMatcher(GpuMatcher):
    alias SIMD_U8_WIDTH = simdwidthof[
        UInt8
    ]()  # // 4  # TODO: needs tuning on wider machines
    alias SIMD_U16_WIDTH = simdwidthof[
        UInt16
    ]()  # // 4  # TODO: needs tuning on wider machines
    var pattern: List[UInt8]
    var rev_pattern: List[UInt8]
    # var rev_haystack_buffer: List[UInt8]
    var profile: Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH]
    var reverse_profile: Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH]
    var matrix: ScoringMatrix

    fn __init__(out self, pattern: List[UInt8]):
        var matrix = ScoringMatrix.all_ascii_default_matrix()
        self.matrix = matrix
        self.pattern = pattern
        self.rev_pattern = create_reversed(pattern)
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

        var result = semi_global_aln_start_end[do_saturation_check=False](
            reference=haystack,
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
        if result and result.value().score >= len(self.pattern):
            return MatchResult(
                Int(result.value().target_start),
                Int(result.value().target_end + 1),
            )

        return None

    fn find_start(
        read self, haystack: Span[UInt8], pattern: Span[UInt8]
    ) -> Int:
        var rev_haystack = create_reversed(haystack)
        var result = semi_global_aln[do_saturation_check=False](
            reference=rev_haystack,
            query_len=len(pattern),
            gap_open_penalty=3,
            gap_extension_penalty=1,
            profile=self.reverse_profile.profile_large.value(),
            bias=self.profile.bias.cast[DType.uint16](),
            max_score=self.profile.max_score,
            min_score=self.profile.min_score,
            free_query_start_gaps=True,
            free_query_end_gaps=True,
            free_target_start_gaps=True,
            free_target_end_gaps=True,
        )
        return len(haystack) - Int(result.best.reference) - 1

    @always_inline
    fn convert_ascii_to_encoding(read self, value: UInt8) -> UInt8:
        """Convert an ascii byte to an encoded byte."""
        return self.matrix.convert_ascii_to_encoding(value)

    @always_inline
    fn convert_encoding_to_ascii(read self, value: UInt8) -> UInt8:
        """Convert an encoded byte to an ascii byte."""
        return self.matrix.convert_encoding_to_ascii(value)

    fn matrix_bytes(read self) -> UnsafePointer[Int8]:
        return self.matrix.values.unsafe_ptr()

    fn matrix_len(read self) -> UInt:
        return len(self.matrix.values)

    @staticmethod
    fn batch_match_coarse[
        max_matrix_length: UInt, max_query_length: UInt, max_target_length: UInt
    ](
        query: DeviceBuffer[DType.uint8],
        ref_buffer: DeviceBuffer[DType.uint8],
        target_ends: DeviceBuffer[DType.uint32],
        score_result_buffer: DeviceBuffer[DType.int32],
        query_end_result_buffer: DeviceBuffer[DType.int32],
        ref_end_result_buffer: DeviceBuffer[DType.int32],
        basic_matrix_values: DeviceBuffer[DType.int8],
        basic_matrix_len: Int,
        query_len: Int,
        target_ends_len: Int,
        thread_count: Int,
    ):
        gpu_align_coarse[
            max_matrix_length, max_query_length, max_target_length
        ](
            query,
            ref_buffer,
            target_ends,
            score_result_buffer,
            query_end_result_buffer,
            ref_end_result_buffer,
            basic_matrix_values,
            basic_matrix_len,
            query_len,
            target_ends_len,
            thread_count,
        )
