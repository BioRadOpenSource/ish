"""Striped Semi-Global Alignment."""

from builtin.math import max
from collections import InlineArray
from math import sqrt, iota
from memory import pack_bits, memset_zero

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
    alias smallVProfile = List[SIMD[SmallType, SIMD_SMALL_WIDTH]]
    alias largeVProfile = List[SIMD[LargeType, SIMD_LARGE_WIDTH]]

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
        var bias: UInt8 = 0
        var max_score: Int8 = 0
        var min_score: Int8 = 0
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
    ) -> List[SIMD[T, size]]:
        """Divide the query into segments."""
        var segment_length = (len(query) + size - 1) // size
        var profile = List[SIMD[T, size]](
            capacity=Int(score_matrix.size * segment_length)
        )
        for _ in range(0, profile.capacity):
            profile.append(SIMD[T, size](0))

        # Generate query profile and rearrange query sequence and calculate the weight of match/mismatch
        var t_idx = 0
        for nt in range(0, score_matrix.size):
            for i in range(0, segment_length):
                var j = i
                for segment_idx in range(0, size):
                    profile[t_idx][segment_idx] = (
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
    dt: DType, width: Int
](
    reference: Span[UInt8],
    rev_reference: Span[UInt8],
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
) -> AlignmentStartEndResult:
    # TODO: use the version with overflow checking?
    # Since saturation happens trivially easily at u8... maybe not.
    var forward = semi_global_aln[dt, width](
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

    var reverse = semi_global_aln[dt, width](
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
        target_start=len(reference) - reverse.best.reference - 1,
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
            gap_open_penalty.cast[SmallType](),
            gap_extension_penalty.cast[SmallType](),
            profile.profile_small.value(),
            profile.bias.cast[SmallType](),
            profile.max_score,
            profile.min_score,
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
            gap_open_penalty.cast[LargeType](),
            gap_extension_penalty.cast[LargeType](),
            profile.profile_large.value(),
            profile.bias.cast[LargeType](),
            profile.max_score,
            profile.min_score,
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
    """Smith-Waterman local alignment.

    Arguments:
        terminate: The best alignment score, used to terminate the matrix calc when locating the alignment beginning point. If this score is set to 0, it will not be used.

    """
    constrained[dt.is_unsigned(), "dt must be signed"]()
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
    # print("POSITION for vposmax:", position)
    var v_pos_mask = SIMD[dt, width](position) == iota[dt, width]().reversed()
    # print("vposmask:", v_pos_mask)

    var score = MIN
    var zero = SIMD[dt, width](ZERO)
    var pv_h_store = List[SIMD[dt, width]](
        capacity=Int(segment_length)
    )  # aka: pvHStore
    # Contains scores from the previous row that will be loaded for calculation
    var pv_h_load = List[SIMD[dt, width]](
        capacity=Int(segment_length)
    )  # aka: pvHLoad
    # Tracks scores for alignments that end with gaps in the query seq (horizontal gaps in visualization)
    var pv_e = List[SIMD[dt, width]](capacity=Int(segment_length))  # aka: pvE
    var boundary = List[SIMD[dt, 1]](capacity=len(reference) + 1)
    memset_zero(pv_h_store.unsafe_ptr(), Int(segment_length))
    memset_zero(pv_h_load.unsafe_ptr(), Int(segment_length))
    memset_zero(pv_e.unsafe_ptr(), Int(segment_length))
    # memset_zero(boundary.unsafe_ptr(), len(reference) + 1)

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
    # print("pv_e and pv_h_store")
    # for i in range(0, segment_length):
    #     print(i, "e:", pv_e[i] - ZERO)
    #     print(i, "h_store:", pv_h_store[i] - ZERO)

    # Init upper boundary
    boundary.append(ZERO)
    for i in range(1, len(reference) + 1):
        var tmp = Int32(ZERO) if free_target_start_gaps else (
            -Int32(gap_open_penalty + (gap_extension_penalty * (i - 1)))
            + Int32(ZERO)
        )
        boundary.append(MIN if tmp < Int32(MIN) else tmp.cast[dt]())

    # print("Boundary:")
    # print()
    # for i in range(0, len(reference) + 1):
    #     print(boundary[i], ", ", end="")
    # print()

    var v_gap_open = SIMD[dt, width](gap_open_penalty)  # aka: vGap0
    var v_gap_ext = SIMD[dt, width](gap_extension_penalty)  # aka: vGapE

    # NEG_LIMIT = (-open < matrix->min ? INT8_MIN + open : INT8_MIN - matrix->min) + 1;
    # POS_LIMIT = INT8_MAX - matrix->max - 1;

    var v_neg_limit: SIMD[dt, width]
    if -Int32(gap_open_penalty) < Int32(min_score):
        v_neg_limit = MIN + gap_open_penalty + 1
    else:
        # this is a gross set of casting to allow the possibly negetive matrix min
        v_neg_limit = (Int32(MIN) + Int32(abs(min_score)) + Int32(1)).cast[dt]()

    var v_pos_limit = SIMD[dt, width](Int32(MAX) - Int32(max_score) - 1)

    # var v_pos_limit = SIMD[dt, width](MAX_LIMIT)
    var v_saturation_check_min = v_neg_limit
    var v_saturation_check_max = v_pos_limit - (v_neg_limit)
    var v_max_h = v_neg_limit
    var v_bias = SIMD[dt, width](bias)

    # 32767
    # Note:
    # H - Score for match / mismatch (diagonal move)
    # E - Score for gap in query (horizontal move)
    # F - Score for gap in reference (vertical move)
    # print("MIN:", MIN)
    # print("MAX:", MAX)
    # print("ZERO:", ZERO)
    # print("bias:", bias)
    # print("Segment length:", segment_length)
    # print("RefLen: ", len(reference))
    # print("querylen: ", query_len)
    # print("SIMD DType", dt)
    # print("SIMD WIDTH", width)

    for i in range(0, len(reference)):  # Not i and j are swapped
        # print("OUTER LOOP:", i)
        var v_e = zero
        var v_f = v_neg_limit
        var v_h = pv_h_store[segment_length - 1]
        v_h = v_h.shift_right[1]()

        # Select the right vector from the query profile
        var profile_idx = reference[i].cast[DType.int32]() * segment_length
        # Swap the two score buffers
        swap(pv_h_load, pv_h_store)

        # Insert the boundary condition
        v_h[0] = boundary[i]

        # print("vH State:", v_h)
        # print("pvHLoad State:", end="")
        # for i in range(0, segment_length):
        #     print(pv_h_load[i], ", ", end="")
        # print()

        # print("pvHStore State:", end="")
        # for i in range(0, segment_length):
        #     print(pv_h_store[i], ", ", end="")
        # print()

        # print("pvE State:", end="")
        # for i in range(0, segment_length):
        #     print(pv_e[i], ", ", end="")
        # print()

        # Inner loop to process the query sequence
        for j in range(0, segment_length):
            # print("\tInner loop for query sequence, checcking segment:", j)
            # print("\tSegement J's query profile:", profile[profile_idx + j])
            # Add profile score to
            v_h = saturating_add(v_h, profile[profile_idx + j])
            v_h = saturating_sub(v_h, v_bias)
            v_e = pv_e[j]
            # print(
            #     "\t\tState of vH after adding profile: ",
            #     v_h,
            # )
            # print(
            #     "\t\tState of vE : ",
            #     v_e,
            # )
            # print(
            #     "\t\tState of vF: ",
            #     v_f,
            # )

            # Get max from current_cell_score, horizontal gap, and vertical gap score
            # print("\t\t", "Get max from current cell / hgap / vgap")
            v_h = max(v_h, v_e)
            v_h = max(v_h, v_f)

            # Save current_cell_score
            pv_h_store[j] = v_h

            @parameter
            if do_saturation_check:
                v_saturation_check_max = max(
                    v_saturation_check_max, v_h - v_neg_limit
                )
                # v_saturation_check_min = min(v_saturation_check_min, v_h)
                v_saturation_check_max = max(
                    v_saturation_check_max, v_e - v_neg_limit
                )
            # v_saturation_check_min = min(v_saturation_check_min, v_e)
            # print("SATCHECK MAX:", v_saturation_check_max)
            # print("SATCHECK MIN:", v_saturation_check_min)

            # update vE
            v_h = saturating_sub(v_h, v_gap_open)
            v_e = saturating_sub(v_e, v_gap_ext)
            v_e = max(v_e, v_h)
            pv_e[j] = v_e
            # print("\t\te State after update: ", v_e)

            # update vF
            v_f = saturating_sub(v_f, v_gap_ext)
            v_f = max(v_f, v_h)
            # print("\t\tvF State after update:", v_f)

            # load the next vH
            v_h = pv_h_load[j]
            # print("\t\tnext vH:", v_h)

        # print("\tStarting LazyF")
        # Lazy_F loop, disallows adjacent insertion and then deletion from SWPS3
        # Possible speedup - check if v_f has any updates to start with
        # var break_out = (v_f == zero).reduce_and()
        # var k = 0
        # while not break_out and k < width:
        # k += 1
        var break_out = False
        for _k in range(0, width):
            var tmp = gap_open_penalty.cast[
                DType.int32
            ]() if free_target_start_gaps else (
                boundary[i + 1] - gap_open_penalty
            ).cast[
                DType.int32
            ]()
            var tmp2 = MIN if tmp < Int32(MIN) else tmp.cast[dt]()

            v_f = v_f.shift_right[1]()
            v_f[0] = tmp2

            # print("\tLeft Shift vF:", v_f)
            # print("\tWalking Segments")
            for j in range(0, segment_length):
                # print("\t\tvF:               ", v_f)
                v_h = pv_h_store[j]
                # print("\t\tvH from store:    ", v_h)
                v_h = max(v_h, v_f)

                pv_h_store[j] = v_h

                @parameter
                if do_saturation_check:
                    v_saturation_check_max = max(
                        v_saturation_check_max, v_h - v_neg_limit
                    )
                # v_saturation_check_min = min(v_saturation_check_min, v_h)
                # print("SATCHECK MAX:", v_saturation_check_max)
                # print("SATCHECK MIN:", v_saturation_check_min)

                v_h = saturating_sub(v_h, v_gap_open)
                v_f = saturating_sub(v_f, v_gap_ext)
                # print("\t\tnew vH: ", v_h)
                # print("\t\tnew vF: ", v_f)

                # Early termination check
                if not (v_f > v_h).reduce_or():
                    break_out = True
                    break
            if break_out:
                break
        # print("\t Done with main loops")

        # Extract vector containing last value from the column
        v_h = pv_h_store[offset]
        # print("\tLast col vector:", v_h)
        # print("\tCurrent v_max_h:", v_max_h)
        # print("vH > vMaxH result:", v_h > v_max_h)
        # print("v_pos_mask:", v_pos_mask)
        var v_compare = v_pos_mask & (v_h > v_max_h)
        # print("\tv_compare:", v_compare)
        v_max_h = max(v_h, v_max_h)
        # print("\tnew v_max_h:", v_max_h)
        if v_compare.reduce_or():
            # print("Setting end_reference to", i)
            end_reference = i

    # Done, handle possible free gaps

    # Max last value from all columns
    if free_target_end_gaps:
        # print("Free target end gaps")
        for _k in range(0, position):
            v_max_h = v_max_h.shift_right[1]()
        score = v_max_h[width - 1]
        end_query = query_len - 1

    # Max of lst column
    if free_query_end_gaps:
        # print("Free query end gaps")
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
        # print("No free end gaps")
        var v_h = pv_h_store[offset]
        # print("\tvH", v_h)
        for _k in range(0, position):
            v_h = v_h.shift_right[1]()
        # print("\tvH post shift", v_h)
        score = v_h[width - 1]
        end_reference = len(reference) - 1
        end_query = query_len - 1

    # Saturation check
    @parameter
    if do_saturation_check:
        if (
            (v_saturation_check_max > v_pos_limit)
            # | (v_saturation_check_min < v_neg_limit)
        ).reduce_or():
            # print(score)
            # print("Saturation")
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
