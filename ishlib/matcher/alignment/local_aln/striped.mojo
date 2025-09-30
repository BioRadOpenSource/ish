"""Striped Smith-Waterman Alignment."""

from benchmark import keep
from builtin.math import max
from collections import InlineArray
from math import sqrt
from memory import pack_bits, memset_zero, memcpy

from ishlib.matcher.alignment import AlignedMemory
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix
from ishlib.matcher.alignment.striped_utils import (
    saturating_sub,
    saturating_add,
    ScoreSize,
    ProfileVectors,
    AlignmentEnd,
    AlignmentResult,
)
from ishlib.vendor.log import Logger


@fieldwise_init
struct Cigar(Copyable, Movable):
    var seq: List[UInt32]


@fieldwise_init
@register_passable
struct ReferenceDirection(ImplicitlyCopyable):
    """Direction of the reference sequence."""

    var value: UInt8
    alias Forward = Self(0)
    alias Reverse = Self(1)

    fn __eq__(read self, read other: Self) -> Bool:
        return self.value == other.value


@fieldwise_init
@register_passable("trivial")
struct Alignment:
    var score1: Int32
    var ref_begin1: Int32
    var ref_end1: Int32
    var read_begin1: Int32
    var read_end1: Int32


@fieldwise_init
struct Profile[SIMD_U8_WIDTH: Int, SIMD_U16_WIDTH: Int](Copyable, Movable):
    alias ByteVProfile = AlignedMemory[
        DType.uint8, SIMD_U8_WIDTH, SIMD_U8_WIDTH
    ]
    alias WordVProfile = AlignedMemory[
        DType.uint16, SIMD_U16_WIDTH, SIMD_U16_WIDTH
    ]

    var profile_byte: Optional[Self.ByteVProfile]
    var profile_word: Optional[Self.WordVProfile]
    var query_len: Int32
    var alphabet_size: UInt32
    var bias: UInt8

    fn __init__(
        out self,
        query: Span[UInt8],
        read score_matrix: ScoringMatrix,
        score_size: ScoreSize,
    ):
        """Query is expected to be converted to range of [0:score_matrix.size].

        i.e. ACTG should be 0, 1, 2, 3
        """
        var profile_byte: Optional[Self.ByteVProfile] = None
        var profile_word: Optional[Self.WordVProfile] = None
        var bias: UInt8 = 0
        if score_size == ScoreSize.Byte or score_size == ScoreSize.Adaptive:
            # Find the bias to use in the substitution matrix
            # The bias will be smallest value in the scoring matrix
            var bias_tmp: Int8 = 0
            for i in range(0, len(score_matrix.values)):
                if score_matrix.values[i] < bias_tmp:
                    bias_tmp = score_matrix.values[i]
            bias = abs(bias_tmp).cast[DType.uint8]()
            profile_byte = Self.generate_query_profile[
                DType.uint8, SIMD_U8_WIDTH
            ](query, score_matrix, bias)

        if score_size == ScoreSize.Word or score_size == ScoreSize.Adaptive:
            # Find the bias to use in the substitution matrix
            # The bias will be smallest value in the scoring matrix
            var bias_tmp: Int8 = 0
            for i in range(0, len(score_matrix.values)):
                if score_matrix.values[i] < bias_tmp:
                    bias_tmp = score_matrix.values[i]
            bias = abs(bias_tmp).cast[DType.uint8]()
            profile_word = Self.generate_query_profile[
                DType.uint16, SIMD_U16_WIDTH
            ](query, score_matrix, bias)

        return Self(
            profile_byte,
            profile_word,
            len(query),
            score_matrix.size,
            bias,
        )

    @staticmethod
    fn generate_query_profile[
        T: DType, size: Int
    ](
        query: Span[UInt8], read score_matrix: ScoringMatrix, bias: UInt8
    ) -> AlignedMemory[T, size, size]:
        """Divide the query into segments."""
        var segment_length = (len(query) + size - 1) // size
        var length = Int(score_matrix.size * segment_length)
        var profile = AlignedMemory[T, size, size](length)

        # Generate query profile and rearrange query sequence and calculate the weight of match/mismatch
        var p = profile.as_span()
        var t_idx = 0
        for nt in range(0, score_matrix.size):
            for i in range(0, segment_length):
                var j = i
                for segment_idx in range(0, size):
                    keep(t_idx)
                    keep(segment_idx)
                    p[t_idx][segment_idx] = (
                        bias if j
                        >= len(query) else (
                            score_matrix.get(nt, Int(query[j]))
                            + bias.cast[DType.int8]()
                        ).cast[DType.uint8]()
                    ).cast[T]()
                    j += segment_length
                t_idx += 1
        return profile^


fn ssw_align[
    SIMD_U8_WIDTH: Int, SIMD_U16_WIDTH: Int
](
    read profile: Profile[SIMD_U8_WIDTH, SIMD_U16_WIDTH],
    read matrix: ScoringMatrix,
    reference: Span[UInt8],
    query: Span[UInt8],
    read reverse_profile: Profile[SIMD_U8_WIDTH, SIMD_U16_WIDTH],
    *,
    gap_open_penalty: UInt8 = 3,
    gap_extension_penalty: UInt8 = 1,
    return_only_alignment_end: Bool = False,
    mask_length: Int32 = 15,  # for second best score
    score_cutoff: Float32 = 0.0,
) -> Optional[Alignment]:
    # Find the alignment scores and ending positions
    var bests: AlignmentResult

    var used_word = False

    if profile.profile_byte:
        bests = sw[DType.uint8, SIMD_U8_WIDTH](
            reference,
            ReferenceDirection.Forward,
            profile.query_len,
            gap_open_penalty,
            gap_extension_penalty,
            profile.profile_byte.value().as_span(),
            -1,
            profile.bias,
            mask_length,
        )
        if (
            profile.profile_word
            and bests.best.score + profile.bias.cast[DType.int32]() >= 255
        ):
            bests = sw[DType.uint16, SIMD_U16_WIDTH](
                reference,
                ReferenceDirection.Forward,
                profile.query_len,
                gap_open_penalty.cast[DType.uint16](),
                gap_extension_penalty.cast[DType.uint16](),
                profile.profile_word.value().as_span(),
                -1,
                profile.bias.cast[DType.uint16](),
                mask_length,
            )
            used_word = True
        elif bests.best.score + profile.bias.cast[DType.int32]() >= 255:
            Logger.warn(
                "Please overflow in alignments detected, please provide a"
                " larger query profile"
            )
            return None

    elif profile.profile_word:
        bests = sw[DType.uint16, SIMD_U16_WIDTH](
            reference,
            ReferenceDirection.Forward,
            profile.query_len,
            gap_open_penalty.cast[DType.uint16](),
            gap_extension_penalty.cast[DType.uint16](),
            profile.profile_word.value().as_span(),
            -1,
            profile.bias.cast[DType.uint16](),
            mask_length,
        )

        used_word = True
    else:
        Logger.warn("Failed to provide a valid query profile")
        return None

    if Float32(bests.best.score) <= score_cutoff:
        Logger.debug("Worse than cutoff")
        return None

    var score1 = bests.best.score
    var ref_end1 = bests.best.reference
    var read_end1 = bests.best.query

    if return_only_alignment_end or ref_end1 <= 0 or read_end1 <= 0:
        return Alignment(
            score1=score1,
            ref_begin1=-1,
            ref_end1=ref_end1,
            read_begin1=-1,
            read_end1=read_end1,
        )

    # Get the start position
    var bests_rev: AlignmentResult
    if not used_word:
        # print("Running rev align")
        bests_rev = sw[DType.uint8, SIMD_U8_WIDTH](
            reference[0 : Int(ref_end1) + 1],
            ReferenceDirection.Reverse,
            reverse_profile.query_len,
            gap_open_penalty,
            gap_extension_penalty,
            reverse_profile.profile_byte.value().as_span(),
            -1,
            reverse_profile.bias,
            mask_length,
        )
    else:
        bests_rev = sw[DType.uint16, SIMD_U16_WIDTH](
            reference[0 : Int(ref_end1) + 1],
            ReferenceDirection.Reverse,
            reverse_profile.query_len,
            gap_open_penalty.cast[DType.uint16](),
            gap_extension_penalty.cast[DType.uint16](),
            reverse_profile.profile_word.value().as_span(),
            -1,
            reverse_profile.bias.cast[DType.uint16](),
            mask_length,
        )
    var ref_begin1 = bests_rev.best.reference
    var read_begin1 = read_end1 - bests_rev.best.query

    # Skipping CIGAR for now
    return Alignment(
        score1=score1,
        ref_begin1=ref_begin1,
        ref_end1=ref_end1,
        read_begin1=read_begin1,
        read_end1=read_end1,
    )


@export
fn sw[
    dt: DType, width: Int
](
    reference: Span[UInt8],
    reference_direction: ReferenceDirection,
    query_len: Int32,
    gap_open_penalty: SIMD[dt, 1],
    gap_extension_penalty: SIMD[dt, 1],
    profile: Span[SIMD[dt, width]],
    # mut p_vecs: ProfileVectors[dt, width],
    terminate: SIMD[dt, 1],
    bias: SIMD[dt, 1],
    mask_length: Int32,
) -> AlignmentResult:
    """Smith-Waterman local alignment.

    Arguments:
        terminate: The best alignment score, used to terminate the matrix calc when locating the alignment beginning point. If this score is set to 0, it will not be used.

    """
    var p_vecs = ProfileVectors[dt, width](query_len)
    p_vecs.init_columns(len(reference))
    var max_score = UInt8(0).cast[dt]()
    var end_query: Int32 = query_len - 1
    var end_reference: Int32 = (
        -1
    )  # 0 based best alignment ending point; initialized as isn't aligned -1
    var segment_length = p_vecs.segment_length

    # Note:
    # H - Score for match / mismatch (diagonal move)
    # E - Score for gap in query (horizontal move)
    # F - Score for gap in reference (vertical move)

    var zero = SIMD[dt, width](0)

    var v_gap_open = SIMD[dt, width](gap_open_penalty)  # aka: vGap0
    var v_gap_ext = SIMD[dt, width](gap_extension_penalty)  # aka: vGapE
    var v_bias = SIMD[dt, width](bias)  # aka: vBias
    # Trace the highest scro of the whole SW matrix
    var v_max_score = zero  # aka: vMaxScore
    # Trace the highest score till the previous column
    var v_max_mark = zero  # aka: vMaxMark

    var begin: Int32 = 0
    var end: Int32 = len(reference)
    var step: Int32 = 1

    # Outer loop to process the reference sequence
    if reference_direction == ReferenceDirection.Reverse:
        begin = len(reference) - 1
        end = -1
        step = -1

    var i = begin
    while i != end:
        # Initialize to 0, any errors in vH will be corrected in lazy_f
        var e: SIMD[dt, width]
        # Represents scores for alignments that end with gaps in the reference seq
        var v_f = zero  # aka: vF
        # the max score in the current column
        var v_max_column = zero  # aka: vMaxColumn
        # The score value currently being calculated
        var v_h = p_vecs.pv_h_store[segment_length - 1]  # aka: vH
        v_h = v_h.shift_right[1]()

        # Select the right vector from the query profile
        var profile_idx = reference[i].cast[DType.int32]() * segment_length

        # Swap the two score buffers
        swap(p_vecs.pv_h_load, p_vecs.pv_h_store)

        # Inner loop to process the query sequence
        for j in range(0, segment_length):
            # Add profile score to
            v_h = saturating_add(v_h, profile[profile_idx + j])
            v_h = saturating_sub(v_h, v_bias)  # adjust for bias

            # Get max from current_cell_score, horizontal gap, and vertical gap score
            e = p_vecs.pv_e[j]
            v_h = max(v_h, e)
            v_h = max(v_h, v_f)
            v_max_column = max(v_max_column, v_h)

            # Save current_cell_score
            p_vecs.pv_h_store[j] = v_h

            # update vE
            v_h = saturating_sub(v_h, v_gap_open)
            e = saturating_sub(e, v_gap_ext)
            e = max(e, v_h)
            p_vecs.pv_e[j] = e

            # update vF
            v_f = saturating_sub(v_f, v_gap_ext)
            v_f = max(v_f, v_h)

            # load the next vH
            v_h = p_vecs.pv_h_load[j]

        # Lazy_F loop, disallows adjacent insertion and then deletion from SWPS3
        var break_out = False

        @parameter
        for _k in range(0, width):
            v_f = v_f.shift_right[1]()
            for j in range(0, segment_length):
                v_h = p_vecs.pv_h_store[j]
                v_h = max(v_h, v_f)
                v_max_column = max(v_max_column, v_h)
                p_vecs.pv_h_store[j] = v_h

                v_h = saturating_sub(v_h, v_gap_open)
                v_f = saturating_sub(v_f, v_gap_ext)

                # Early termination check
                if not v_f.gt(v_h).reduce_or():
                    break_out = True
                    break
            if break_out:
                break

        # Check for new max score
        v_max_score = max(v_max_score, v_max_column)
        var equal_vector = v_max_mark.eq(v_max_score)
        if not equal_vector.reduce_and():
            # find max score in vector
            var temp = v_max_score.reduce_max()
            v_max_mark = v_max_score

            if temp > max_score:
                max_score = temp
                if (max_score + bias) >= SIMD[dt, 1].MAX:
                    break
                end_reference = i

                # Store the column with the highest alignment score in order to trace teh alignment ending position on the query
                for j in range(0, segment_length):
                    p_vecs.pv_h_max[j] = p_vecs.pv_h_store[j]

        # Record the max score of current column
        p_vecs.max_column[i] = v_max_column.reduce_max()
        if p_vecs.max_column[i] >= terminate:
            break

        # Increment the while loop
        i += step

    # Trace the alignment ending position on query
    # var column_len = segment_length * SIMD_U8_WIDTH
    for i in range(0, segment_length):
        for j in range(0, width):
            if p_vecs.pv_h_max[i][j] == max_score:
                var temp = i + j * segment_length
                if temp < Int(end_query):
                    end_query = temp

    var bests = AlignmentResult(
        AlignmentEnd(max_score.cast[DType.int32](), end_reference, end_query),
    )

    return bests
