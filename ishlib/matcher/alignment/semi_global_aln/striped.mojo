"""Striped Semi-Global Alignment."""

from benchmark import keep
from builtin.math import max
from collections import InlineArray
from math import sqrt, iota
from memory import pack_bits, memset_zero

from ishlib.matcher.alignment import create_reversed, AlignedMemory
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix
from ishlib.matcher.alignment.striped_utils import (
    saturating_sub,
    saturating_add,
    ScoreSize,
    ProfileVectors,
    AlignmentEnd,
    AlignmentResult,
    AlignmentStartEndResult,
)


@value
struct Profile[
    SIMD_SMALL_WIDTH: Int,
    SIMD_LARGE_WIDTH: Int,
    SmallType: DType = DType.uint8,
    LargeType: DType = DType.uint16,
]:
    alias smallVProfile = AlignedMemory[
        SmallType, SIMD_SMALL_WIDTH, SIMD_SMALL_WIDTH
    ]
    alias largeVProfile = AlignedMemory[
        LargeType, SIMD_LARGE_WIDTH, SIMD_LARGE_WIDTH
    ]

    var profile_small: Optional[Self.smallVProfile]
    var profile_large: Optional[Self.largeVProfile]
    var query_len: Int32
    var alphabet_size: UInt32
    var bias: UInt8
    var max_score: Int8
    var min_score: Int8

    fn __init__(
        out self,
        query: Span[UInt8],
        read score_matrix: ScoringMatrix,
        score_size: ScoreSize,
    ):
        """Query is expected to be converted to range of [0:score_matrix.size].

        i.e. ACTG should be 0, 1, 2, 3
        """
        constrained[SmallType.is_unsigned(), "SmallType must be unsigned"]()
        constrained[LargeType.is_unsigned(), "LargeType must be unsigned"]()
        # print("Intializing a profile for a query of len", len(query))
        var profile_small: Optional[Self.smallVProfile] = None
        var profile_large: Optional[Self.largeVProfile] = None
        var bias: UInt8
        var max_score: Int8
        var min_score: Int8
        # Find the bias to use in the substitution matrix
        # The bias will be smallest value in the scoring matrix
        var bias_tmp: Int8 = 0
        var min_tmp: Int8 = 0
        var max_tmp: Int8 = 0
        for i in range(0, len(score_matrix.values)):
            var score = score_matrix.values[i]
            if score < bias_tmp:
                bias_tmp = score_matrix.values[i]
            if score < min_tmp:
                min_tmp = score
            if score > max_tmp:
                max_tmp = score
        bias = abs(bias_tmp).cast[DType.uint8]()
        min_score = min_tmp
        max_score = max_tmp

        if score_size == ScoreSize.Byte or score_size == ScoreSize.Adaptive:
            profile_small = Self.generate_query_profile[
                SmallType, SIMD_SMALL_WIDTH
            ](query, score_matrix, bias)

        if score_size == ScoreSize.Word or score_size == ScoreSize.Adaptive:
            profile_large = Self.generate_query_profile[
                LargeType, SIMD_LARGE_WIDTH
            ](query, score_matrix, bias)

        return Self(
            profile_small,
            profile_large,
            len(query),
            score_matrix.size,
            bias,
            max_score,
            min_score,
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
        return profile


fn semi_global_aln_start_end[
    dt: DType, width: Int, *, do_saturation_check: Bool = True
](
    reference: Span[UInt8],
    query_len: Int32,
    gap_open_penalty: SIMD[dt, 1],
    gap_extension_penalty: SIMD[dt, 1],
    profile: Span[SIMD[dt, width]],
    rev_profile: Span[SIMD[dt, width]],
    bias: Scalar[dt],
    max_score: Int8,
    min_score: Int8,
    *,
    free_query_start_gaps: Bool = False,
    free_query_end_gaps: Bool = False,
    free_target_start_gaps: Bool = False,
    free_target_end_gaps: Bool = False,
    score_cutoff: Int32 = 1,
) -> Optional[AlignmentStartEndResult]:
    # TODO: use the version with overflow checking?
    # Since saturation happens trivially easily at u8... maybe not.
    var forward = semi_global_aln[
        dt, width, do_saturation_check=do_saturation_check
    ](
        reference,
        query_len,
        gap_open_penalty,
        gap_extension_penalty,
        profile,
        bias,
        max_score,
        min_score,
        free_query_start_gaps=free_query_start_gaps,
        free_query_end_gaps=free_query_end_gaps,
        free_target_start_gaps=free_target_start_gaps,
        free_target_end_gaps=free_target_end_gaps,
    )

    if forward.best.score < score_cutoff:
        return None

    var rev_reference = create_reversed(
        reference[0 : Int(forward.best.reference) + 1]
    )
    var reverse = semi_global_aln[
        dt, width, do_saturation_check=do_saturation_check
    ](
        rev_reference,
        query_len,
        gap_open_penalty,
        gap_extension_penalty,
        rev_profile,
        bias,
        max_score,
        min_score,
        # Note the swapped free gaps
        free_query_start_gaps=free_query_end_gaps,
        free_query_end_gaps=free_query_start_gaps,
        free_target_start_gaps=free_target_end_gaps,
        free_target_end_gaps=free_target_start_gaps,
    )

    return AlignmentStartEndResult(
        score=forward.best.score,
        query_start=query_len - reverse.best.query - 1,
        query_end=forward.best.query,
        target_start=len(rev_reference) - reverse.best.reference - 1,
        target_end=forward.best.reference,
    )


fn semi_global_aln_with_saturation_check[
    SIMD_SMALL_WIDTH: Int,
    SIMD_LARGE_WIDTH: Int,
    SmallType: DType = DType.uint8,
    LargeType: DType = DType.uint16,
](
    reference: Span[UInt8],
    query_len: Int32,
    gap_open_penalty: SIMD[LargeType, 1],
    gap_extension_penalty: SIMD[LargeType, 1],
    profile: Profile[SIMD_SMALL_WIDTH, SIMD_LARGE_WIDTH, SmallType, LargeType],
    *,
    free_query_start_gaps: Bool = False,
    free_query_end_gaps: Bool = False,
    free_target_start_gaps: Bool = False,
    free_target_end_gaps: Bool = False,
    score_size: ScoreSize = ScoreSize.Adaptive,
) -> AlignmentResult:
    var result: AlignmentResult = AlignmentResult(AlignmentEnd(0, 0, 0))

    if score_size == ScoreSize.Adaptive or score_size == ScoreSize.Byte:
        result = semi_global_aln[SmallType, SIMD_SMALL_WIDTH](
            reference,
            query_len,
            gap_open_penalty=gap_open_penalty.cast[SmallType](),
            gap_extension_penalty=gap_extension_penalty.cast[SmallType](),
            profile=profile.profile_small.value().as_span(),
            bias=profile.bias.cast[SmallType](),
            max_score=profile.max_score,
            min_score=profile.min_score,
            free_query_start_gaps=free_query_start_gaps,
            free_query_end_gaps=free_query_end_gaps,
            free_target_start_gaps=free_target_start_gaps,
            free_target_end_gaps=free_target_end_gaps,
        )
    var overflow_detected = result.overflow_detected
    if (
        score_size == ScoreSize.Adaptive and overflow_detected
    ) or score_size == ScoreSize.Word:
        var larger = semi_global_aln[LargeType, SIMD_LARGE_WIDTH](
            reference,
            query_len,
            gap_open_penalty=gap_open_penalty.cast[LargeType](),
            gap_extension_penalty=gap_extension_penalty.cast[LargeType](),
            profile=profile.profile_large.value().as_span(),
            bias=profile.bias.cast[LargeType](),
            max_score=profile.max_score,
            min_score=profile.min_score,
            free_query_start_gaps=free_query_start_gaps,
            free_query_end_gaps=free_query_end_gaps,
            free_target_start_gaps=free_target_start_gaps,
            free_target_end_gaps=free_target_end_gaps,
        )
        if not larger.overflow_detected and overflow_detected:
            larger.overflow_detected = overflow_detected
        result = larger

    return result


fn semi_global_aln[
    dt: DType, width: Int, *, do_saturation_check: Bool = True
](
    reference: Span[UInt8],
    query_len: Int32,
    gap_open_penalty: SIMD[dt, 1],
    gap_extension_penalty: SIMD[dt, 1],
    profile: Span[SIMD[dt, width]],
    bias: Scalar[dt],
    max_score: Int8,
    min_score: Int8,
    *,
    free_query_start_gaps: Bool = False,
    free_query_end_gaps: Bool = False,
    free_target_start_gaps: Bool = False,
    free_target_end_gaps: Bool = False,
) -> AlignmentResult:
    """Semi-global alignment."""
    constrained[dt.is_unsigned(), "dt must be unsigned"]()
    alias NUM = Scalar[dt]
    alias MIN = NUM.MIN_FINITE
    alias MAX = NUM.MAX_FINITE
    alias ZERO = MAX // 2  # we are pretending to be a signed int and just shifting up.

    if query_len == 0 or len(reference) == 0:
        return AlignmentResult(AlignmentEnd(0, 0, 0))

    var end_query: Int32 = query_len - 1
    var end_reference: Int32 = -1  # 0 based best alignment ending point; initialized as isn't aligned -1
    var segment_length = (query_len + width - 1) // width
    var offset = (query_len - 1) % segment_length
    var position = (width - 1) - (query_len - 1) // segment_length
    var v_pos_mask = SIMD[dt, width](position) == iota[dt, width]().reversed()

    var score = MIN
    var zero = SIMD[dt, width](ZERO)
    var pv_h_store = AlignedMemory[dt, width, width](Int(segment_length))
    # Contains scores from the previous row that will be loaded for calculation
    var pv_h_load = AlignedMemory[dt, width, width](Int(segment_length))
    # Tracks scores for alignments that end with gaps in the query seq (horizontal gaps in visualization)
    var pv_e = AlignedMemory[dt, width, width](Int(segment_length))
    var boundary = List[SIMD[dt, 1]](capacity=len(reference) + 1)

    # TODO: better handling for overflows on setup, just return a new return type.
    # Init H and E
    var index = 0
    for i in range(0, segment_length):
        var h = zero
        var e = zero
        for seg_num in range(0, width):
            var tmp = Int32(ZERO) if free_query_start_gaps else (
                Int32(ZERO)
                - (
                    Int32(gap_open_penalty)
                    + Int32(gap_extension_penalty)
                    * (seg_num * segment_length + i)
                )
            ).cast[DType.int32]()
            h[seg_num] = MIN if tmp < Int32(MIN) else tmp.cast[dt]()
            tmp = tmp - Int32(gap_open_penalty)
            e[seg_num] = MIN if tmp < Int32(MIN) else tmp.cast[dt]()
        pv_h_store[index] = h
        pv_e[index] = e
        index += 1

    # Init upper boundary
    boundary.append(ZERO)
    for i in range(1, len(reference) + 1):
        var tmp = Int32(ZERO) if free_target_start_gaps else (
            -Int32(gap_open_penalty + (gap_extension_penalty * (i - 1)))
            + Int32(ZERO)
        )
        boundary.append(MIN if tmp < Int32(MIN) else tmp.cast[dt]())

    var v_gap_open = SIMD[dt, width](gap_open_penalty)  # aka: vGap0
    var v_gap_ext = SIMD[dt, width](gap_extension_penalty)  # aka: vGapE

    var v_neg_limit: SIMD[dt, width]
    var v_neg_limit_saturation: SIMD[dt, width]
    if -Int32(gap_open_penalty) < Int32(min_score):
        v_neg_limit = MIN + gap_open_penalty + 1
        v_neg_limit_saturation = MAX - gap_open_penalty - 1
    else:
        # this is a gross set of casting to allow the possibly negative matrix min
        v_neg_limit = (Int32(MIN) + Int32(abs(min_score)) + Int32(1)).cast[dt]()
        v_neg_limit_saturation = (
            Int32(MAX) - Int32(abs(min_score)) - Int32(1)
        ).cast[dt]()

    var v_pos_limit = SIMD[dt, width](Int32(MAX) - Int32(max_score) - 1)

    var v_saturation_check_max = min(v_pos_limit, v_neg_limit_saturation)
    var v_saturation_check_max_reference = v_saturation_check_max
    var v_max_h = v_neg_limit
    var v_bias = SIMD[dt, width](bias)

    # Note:
    # H - Score for match / mismatch (diagonal move)
    # E - Score for gap in query (horizontal move)
    # F - Score for gap in reference (vertical move)
    for i in range(0, len(reference)):  # Not i and j are swapped
        var v_e: SIMD[dt, width]
        var v_f = v_neg_limit
        var v_h = pv_h_store[segment_length - 1]
        v_h = v_h.shift_right[1]()

        # Select the right vector from the query profile
        var profile_idx = reference[i].cast[DType.int32]() * segment_length
        # Swap the two score buffers
        swap(pv_h_load, pv_h_store)

        # Insert the boundary condition
        v_h[0] = boundary[i]

        # Inner loop to process the query sequence
        for j in range(0, segment_length):
            # Add profile score to
            v_h = saturating_add(v_h, profile[profile_idx + j])
            v_h = saturating_sub(v_h, v_bias)
            v_e = pv_e[j]

            # Get max from current_cell_score, horizontal gap, and vertical gap score
            v_h = max(v_h, v_e)
            v_h = max(v_h, v_f)

            # Save current_cell_score
            pv_h_store[j] = v_h

            @parameter
            if do_saturation_check:
                v_saturation_check_max = max(
                    v_saturation_check_max, v_h - v_neg_limit
                )
                v_saturation_check_max = max(
                    v_saturation_check_max, v_e - v_neg_limit
                )

            # update vE
            v_h = saturating_sub(v_h, v_gap_open)
            v_e = saturating_sub(v_e, v_gap_ext)
            v_e = max(v_e, v_h)
            pv_e[j] = v_e

            # update vF
            v_f = saturating_sub(v_f, v_gap_ext)
            v_f = max(v_f, v_h)

            # load the next vH
            v_h = pv_h_load[j]

        # Lazy_F loop, disallows adjacent insertion and then deletion from SWPS3
        # Possible speedup - check if v_f has any updates to start with
        var break_out = False
        for _k in range(0, width):
            var tmp = (ZERO - gap_open_penalty).cast[
                DType.int32
            ]() if free_target_start_gaps else (
                boundary[i + 1] - gap_open_penalty
            ).cast[
                DType.int32
            ]()
            var tmp2 = MIN if tmp < Int32(MIN) else tmp.cast[dt]()

            v_f = v_f.shift_right[1]()
            v_f[0] = tmp2

            for j in range(0, segment_length):
                v_h = pv_h_store[j]
                v_h = max(v_h, v_f)

                pv_h_store[j] = v_h

                @parameter
                if do_saturation_check:
                    v_saturation_check_max = max(
                        v_saturation_check_max, v_h - v_neg_limit
                    )

                v_h = saturating_sub(v_h, v_gap_open)
                v_f = saturating_sub(v_f, v_gap_ext)

                # Early termination check
                if not (v_f > v_h).reduce_or():
                    break_out = True
                    break
            if break_out:
                break

        # Extract vector containing last value from the column
        v_h = pv_h_store[offset]
        var v_compare = v_pos_mask & (v_h > v_max_h)
        v_max_h = max(v_h, v_max_h)
        if v_compare.reduce_or():
            end_reference = i

    # Done, handle possible free gaps

    # Max last value from all columns
    if free_target_end_gaps:
        for _k in range(0, position):
            v_max_h = v_max_h.shift_right[1]()
        score = v_max_h[width - 1]
        end_query = query_len - 1

    # Max of lst column
    if free_query_end_gaps:
        for i in range(0, segment_length):
            for j in range(0, width):
                var temp = i + j * segment_length
                if temp >= query_len:
                    continue
                if pv_h_store[i][j] > score:
                    score = pv_h_store[i][j]
                    end_query = temp
                    end_reference = len(reference) - 1
                elif (
                    pv_h_store[i][j] == score
                    and end_reference == len(reference) - 1
                    and temp < end_query
                ):
                    end_query = temp

    # Extract last value from the last column
    if not free_target_end_gaps and not free_query_end_gaps:
        var v_h = pv_h_store[offset]
        for _k in range(0, position):
            v_h = v_h.shift_right[1]()
        score = v_h[width - 1]
        end_reference = len(reference) - 1
        end_query = query_len - 1

    # Saturation check
    @parameter
    if do_saturation_check:
        if (
            (v_saturation_check_max > v_saturation_check_max_reference)
        ).reduce_or():
            return AlignmentResult(
                AlignmentEnd(0, 0, 0), overflow_detected=True
            )

    var bests = AlignmentResult(
        AlignmentEnd(
            score.cast[DType.int32]() - ZERO.cast[DType.int32](),
            end_reference,
            end_query,
        ),
    )

    return bests
