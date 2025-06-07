"""Smith-Waterman local alignment."""
from gpu.host import DeviceContext
from sys.info import simdwidthof

from ishlib.gpu.kernels.semi_global import gpu_align_coarse
from ishlib.matcher import Matcher, MatchResult
from ishlib.matcher.alignment import create_reversed
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix, MatrixKind
from ishlib.matcher.alignment.semi_global_aln.striped import (
    Profile,
    ScoreSize,
    semi_global_aln_start_end,
    semi_global_aln,
)
from ishlib.searcher_settings import SemiGlobalEndsFreeness
from ishlib.vendor.log import Logger


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
    var max_score: Int
    # var rev_haystack_buffer: List[UInt8]
    var profile: Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH]
    var reverse_profile: Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH]
    var matrix: ScoringMatrix
    var _matrix_kind: MatrixKind
    var gap_open: UInt
    var gap_extend: UInt
    var _score_threshold: Float32
    var sg_ends_free: SemiGlobalEndsFreeness

    fn __init__(
        out self,
        pattern: List[UInt8],
        score_threshold: Float32,
        sg_ends_free: SemiGlobalEndsFreeness,
        matrix_kind: MatrixKind = MatrixKind.ASCII,
        gap_open: UInt = 3,
        gap_extend: UInt = 1,
    ):
        Logger.info("Performing matching with StripedSemiGlobalMatcher.")
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self._score_threshold = score_threshold
        self.matrix = matrix_kind.matrix()
        self._matrix_kind = matrix_kind
        self.sg_ends_free = sg_ends_free
        (
            self.pattern,
            self.max_score,
        ) = self.matrix.convert_ascii_to_encoding_and_score(pattern)
        self.rev_pattern = create_reversed(self.pattern)
        var profile = Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH](
            self.pattern, self.matrix, ScoreSize.Word
        )
        self.profile = profile
        var reverse_profile = Profile[Self.SIMD_U8_WIDTH, Self.SIMD_U16_WIDTH](
            self.rev_pattern, self.matrix, ScoreSize.Word
        )
        self.reverse_profile = reverse_profile

    fn first_match(
        read self, haystack: Span[UInt8], _pattern: Span[UInt8]
    ) -> Optional[MatchResult]:
        """Find the first match in the haystack."""

        var result = semi_global_aln_start_end[do_saturation_check=False](
            reference=haystack,
            query_len=len(self.pattern),
            gap_open_penalty=self.gap_open,
            gap_extension_penalty=self.gap_extend,
            profile=self.profile.profile_large.value().as_span(),
            rev_profile=self.reverse_profile.profile_large.value().as_span(),
            bias=self.profile.bias.cast[DType.uint16](),
            max_score=self.profile.max_score,
            min_score=self.profile.min_score,
            free_query_start_gaps=self.sg_ends_free.query_start,
            free_query_end_gaps=self.sg_ends_free.query_end,
            free_target_start_gaps=self.sg_ends_free.target_start,
            free_target_end_gaps=self.sg_ends_free.target_end,
            score_cutoff=Int32(len(self.pattern)),
        )
        if (
            result
            and Float32(result.value().score) / self.max_alignment_score()
            >= self.score_threshold()
        ):
            return MatchResult(
                Int(result.value().target_start),
                Int(result.value().target_end + 1),
            )

        return None

    fn find_start(
        read self, haystack: Span[UInt8], _pattern: Span[UInt8]
    ) -> Int:
        var rev_haystack = create_reversed(haystack)
        var result = semi_global_aln[do_saturation_check=False](
            reference=rev_haystack,
            query_len=len(self.pattern),
            gap_open_penalty=self.gap_open,
            gap_extension_penalty=self.gap_extend,
            profile=self.reverse_profile.profile_large.value().as_span(),
            bias=self.profile.bias.cast[DType.uint16](),
            max_score=self.profile.max_score,
            min_score=self.profile.min_score,
            free_query_start_gaps=self.sg_ends_free.query_end,
            free_query_end_gaps=self.sg_ends_free.query_start,
            free_target_start_gaps=self.sg_ends_free.target_end,
            free_target_end_gaps=self.sg_ends_free.target_start,
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

    @always_inline
    fn matrix_bytes(read self) -> UnsafePointer[Int8]:
        return self.matrix.values.unsafe_ptr()

    @always_inline
    fn matrix_len(read self) -> UInt:
        return len(self.matrix.values)

    @always_inline
    fn matrix_kind(read self) -> MatrixKind:
        return self._matrix_kind

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
        """Returns the score threshold needed to be considered a match."""
        return self._score_threshold

    @staticmethod
    fn batch_match_coarse[
        max_query_length: UInt,
        max_target_length: UInt,
    ](
        query: UnsafePointer[Scalar[DType.uint8]],
        ref_buffer: UnsafePointer[Scalar[DType.uint8]],
        target_ends: UnsafePointer[Scalar[DType.uint32]],
        score_result_buffer: UnsafePointer[Scalar[DType.int32]],
        query_end_result_buffer: UnsafePointer[Scalar[DType.int32]],
        ref_end_result_buffer: UnsafePointer[Scalar[DType.int32]],
        basic_matrix_values: UnsafePointer[Scalar[DType.int8]],
        basic_matrix_len: Int,
        matrix_kind: MatrixKind,
        query_len: Int,
        target_ends_len: Int,
        thread_count: Int,
        gap_open: UInt,
        gap_extend: UInt,
        ends_free: SemiGlobalEndsFreeness,
    ):
        # TODO: make this a param?
        if matrix_kind == MatrixKind.ASCII:
            gpu_align_coarse[
                MatrixKind.ASCII, max_query_length, max_target_length
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
                gap_open,
                gap_extend,
                ends_free.query_start,
                ends_free.query_end,
                ends_free.target_start,
                ends_free.target_end,
            )
        elif matrix_kind == MatrixKind.ACTGN:
            gpu_align_coarse[
                MatrixKind.ACTGN, max_query_length, max_target_length
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
                gap_open,
                gap_extend,
                ends_free.query_start,
                ends_free.query_end,
                ends_free.target_start,
                ends_free.target_end,
            )
        elif matrix_kind == MatrixKind.ACTGN0:
            gpu_align_coarse[
                MatrixKind.ACTGN0, max_query_length, max_target_length
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
                gap_open,
                gap_extend,
                ends_free.query_start,
                ends_free.query_end,
                ends_free.target_start,
                ends_free.target_end,
            )
        elif matrix_kind == MatrixKind.BLOSUM62:
            gpu_align_coarse[
                MatrixKind.BLOSUM62, max_query_length, max_target_length
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
                gap_open,
                gap_extend,
                ends_free.query_start,
                ends_free.query_end,
                ends_free.target_start,
                ends_free.target_end,
            )
        else:
            Logger.warn(
                "No valid MatrixKind set for gpu_align_coarse: ",
                String(matrix_kind),
            )
            gpu_align_coarse[
                MatrixKind.ASCII, max_query_length, max_target_length
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
                gap_open,
                gap_extend,
                ends_free.query_start,
                ends_free.query_end,
                ends_free.target_start,
                ends_free.target_end,
            )
