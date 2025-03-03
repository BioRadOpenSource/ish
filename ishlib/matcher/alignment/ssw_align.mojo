"""Striped Smith-Waterman Alignment."""

from builtin.math import max
from collections import InlineArray
from memory import pack_bits
from sys.intrinsics import likely, unlikely
from sys.info import simdwidthof

alias SIMD_U8_WIDTH = simdwidthof[UInt8]()
alias SIMD_U16_WIDTH = simdwidthof[UInt16]()

alias NUM_TO_NT = InlineArray[UInt8, 5](
    ord("A"), ord("C"), ord("G"), ord("T"), ord("N")
)
"""Table to convert an Int8 to the ascii value of a nucleotide"""
# This table is used to transform nucleotide letters into numbers.
alias NT_TO_NUM = InlineArray[UInt8, 128](
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    0,
    4,
    1,
    4,
    4,
    4,
    2,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    3,
    0,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    0,
    4,
    1,
    4,
    4,
    4,
    2,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    3,
    0,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
)
"""Table used to transform nucleotide letters into numbers."""


@always_inline
fn nt_to_num(owned seq: List[Int8]) -> List[UInt8]:
    var out = List[UInt8](capacity=len(seq))
    for value in seq:
        out.append(NT_TO_NUM[Int(value[])])
    return out


fn nt_to_num(owned seq: List[UInt8]) -> List[UInt8]:
    for value in seq:
        value[] = NT_TO_NUM[Int(value[])]
    return seq


@always_inline
fn num_to_nt(mut seq: List[UInt8]):
    for value in seq:
        value[] = NUM_TO_NT[Int(value[])]


@value
struct ScoringMatrix:
    # TODO: force this to be an inline list that can be copied onto the profile?
    var values: List[Int8]
    var size: UInt32  # The number of values represented (which is the length of an edge of the square matrix)

    fn set_last_row_to_value(mut self, value: Int8 = 2):
        for i in range((self.size - 1) * self.size, len(self.values)):
            self.values[i] = value

    @staticmethod
    fn default_matrix(
        size: UInt32, matched: Int8 = 2, mismatched: Int8 = -2
    ) -> Self:
        var values = List[Int8](capacity=Int(size * size))
        for _ in range(0, Int(size * size)):
            values.append(0)
        for i in range(0, size):
            for j in range(0, size):
                if i == j:
                    values[i * size + j] = matched  # match
                else:
                    values[i * size + j] = mismatched  # Mismatch
        return Self(values, size)

    @staticmethod
    fn actg_default_matrix() -> Self:
        return Self.default_matrix(4)

    @staticmethod
    fn actgn_default_matrix() -> Self:
        return Self.default_matrix(5)

    @staticmethod
    fn all_ascii_default_matrix() -> Self:
        return Self.default_matrix(256)

    fn get(read self, i: Int, j: Int) -> Int8:
        return self.values[i * self.size + j]


@value
struct Cigar:
    var seq: List[UInt32]


@value
struct ReferenceDirection:
    """Direction of the reference sequence."""

    var value: UInt8
    alias Forward = Self(0)
    alias Reverse = Self(1)

    fn __eq__(read self, read other: Self) -> Bool:
        return self.value == other.value


@value
struct ScoreSize:
    """Controls the precision used for alignment scores.

    This directly effects the max possible score.
    """

    var value: UInt8

    alias Byte = Self(0)
    """Use only 8-bit (byte) precision for scores."""
    alias Word = Self(1)
    """Use only 16-bit (word) precision for scores."""
    alias Adaptive = Self(2)
    """Use both precisions, adaptive strategy."""

    fn __eq__(read self, read other: Self) -> Bool:
        return self.value == other.value


@value
struct Profile[
    mut: Bool, //,
    query_origin: Origin[mut],
    score_matrix_origin: ImmutableOrigin,
]:
    alias ByteVProfile = List[SIMD[DType.uint8, SIMD_U8_WIDTH]]
    alias WordVProfile = List[SIMD[DType.uint16, SIMD_U16_WIDTH]]

    var profile_byte: Optional[Self.ByteVProfile]
    var profile_word: Optional[Self.WordVProfile]
    var query: Span[UInt8, query_origin]  # changing "read" to query
    var scoring_matrix: Pointer[ScoringMatrix, score_matrix_origin]
    var query_len: Int32
    var alphabet_size: UInt32
    var bias: UInt8

    fn __init__(
        out self,
        query: Span[UInt8, query_origin],
        ref [score_matrix_origin]score_matrix: ScoringMatrix,
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
            profile_word = Self.generate_query_profile[
                DType.uint16, SIMD_U16_WIDTH
            ](query, score_matrix, bias)

        return Self(
            profile_byte,
            profile_word,
            query,
            Pointer.address_of(score_matrix),
            len(query),
            score_matrix.size,
            bias,
        )

    @staticmethod
    fn generate_query_profile[
        T: DType, size: Int = simdwidthof[T]()
    ](
        query: Span[UInt8], read score_matrix: ScoringMatrix, bias: UInt8
    ) -> List[SIMD[T, size]]:
        """Divide the query into segments."""

        var segment_length = (len(query) + size - 1) // size
        # for i in range(0, len(score_matrix.values)):
        #     print(score_matrix.values[i])
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
                    # if j >= len(query):
                    #     profile[t_idx][segment_idx] = bias.cast[T]()
                    # else:
                    #     profile[t_idx][segment_idx] = (
                    #         score_matrix.get(nt, Int(query[j]))
                    #         + bias.cast[DType.int8]()
                    #     ).cast[T]()
                    # print(profile[t_idx][segment_idx])
                    # ^ non-ternary version
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


@value
struct Alignment:
    var score1: UInt16
    var score2: UInt16
    var ref_begin1: Int32
    var ref_end1: Int32
    var ref_begin2: Int32
    var ref_end2: Int32
    var read_begin1: Int32
    var read_end1: Int32
    var read_begin2: Int32
    var read_end2: Int32
    var cigar: List[UInt32]
    var flag: UInt16


@value
@register_passable("trivial")
struct AlignmentEnd:
    var score: UInt16
    var reference: Int32  # 0-based
    var query: Int32  # alignment ending position on query, 0-based


@always_inline
fn saturating_sub[
    data: DType, width: Int
](lhs: SIMD[data, width], rhs: SIMD[data, width]) -> SIMD[data, width]:
    """Saturating SIMD subtraction.

    https://stackoverflow.com/questions/33481295/saturating-subtract-add-for-unsigned-bytes
    """
    constrained[data.is_unsigned()]()
    var resp = lhs - rhs
    resp &= -(resp <= lhs).cast[data]()
    return resp


@always_inline
fn saturating_add[
    data: DType, width: Int
](lhs: SIMD[data, width], rhs: SIMD[data, width]) -> SIMD[data, width]:
    """Saturating SIMD subtraction.

    https://stackoverflow.com/questions/33481295/saturating-subtract-add-for-unsigned-bytes
    """
    constrained[data.is_unsigned()]()
    var resp = lhs + rhs
    resp |= -(resp < lhs).cast[data]()
    return resp


fn ssw_align[
    profile_origin: ImmutableOrigin
](
    read profile: Profile,
    reference: Span[UInt8],
    *,
    gap_open_penalty: UInt8 = 3,
    gap_extension_penalty: UInt8 = 1,
    settings_flag: UInt8 = 0,
    filter_score: UInt16 = 0,
    filter_distance: Int32 = 0,
    mask_length: UInt32 = 15,  # for second best score
) -> Alignment:
    # Find the alignment scores and ending positions
    var bests: List[AlignmentEnd]
    if profile.profile_byte:
        bests = sw_byte(
            reference,
            ReferenceDirection.Forward,
            profile.query_len,
            gap_open_penalty,
            gap_extension_penalty,
            profile.profile_byte.value(),
            255,
            profile.bias,
            mask_length,
        )
        if profile.profile_word and bests[0].score == 255:
            bests.clear()
            # TODO: run word version
    elif profile.profile_word:
        bests = List[AlignmentEnd]()
        # TODO
    else:
        pass
        # TODO explode

    pass


fn sw_byte(
    reference: Span[UInt8],
    reference_direction: ReferenceDirection,
    query_len: Int32,
    gap_open_penalty: UInt8,
    gap_extension_penalty: UInt8,
    profile: Span[SIMD[DType.uint8, SIMD_U8_WIDTH]],
    terminate: UInt8,
    bias: UInt8,
    mask_length: UInt32,
) -> List[AlignmentEnd]:
    """Smith-Waterman local alignment.

    Arguments:
        terminate: The best alignment score, used to terminate the matrix calc when locating the alignment beginning point. If this score is set to 0, it will not be used.

    """

    var max_score: UInt8 = 0
    var end_query: Int32 = query_len - 1
    var end_reference: Int32 = -1  # 0 based best alignment ending point; initialized as isn't aligned -1
    var segment_length = (query_len + SIMD_U8_WIDTH - 1) // SIMD_U8_WIDTH

    # Note:
    # H - Score for match / mismatch (diagonal move)
    # E - Score for gap in query (horizontal move)
    # F - Score for gap in reference (vertical move)
    print("Segment length:", segment_length)
    print("Byte Profile")
    for i in range(0, 5):  # TODO: hardcoded, should be length alphabet
        print(chr(Int(NUM_TO_NT[i])), ": ", sep="", end="")
        for j in range(0, segment_length):
            print(profile[i * segment_length + j], ", ", end="")
        print()

    # List to record the largest score of each reference position
    var max_column = List[UInt8](capacity=len(reference))
    # List to record the alignment query ending position of the largest score of each reference position
    var end_query_column = List[Int32](capacity=len(reference))
    for _ in range(0, len(reference)):
        max_column.append(0)
        end_query_column.append(0)

    var zero = SIMD[DType.uint8, SIMD_U8_WIDTH](0)
    # Note: I'm using the `pv` prefix from the source even though its irrelevant here, it means `pointer` `vector`
    # Stores the H values (scores) for the current row of the dynamic programming matrix
    var pv_h_store = List[SIMD[DType.uint8, SIMD_U8_WIDTH]](
        capacity=Int(segment_length)
    )  # aka: pvHStore
    # Contains scores from the previous row that will be loaded for calculation
    var pv_h_load = List[SIMD[DType.uint8, SIMD_U8_WIDTH]](
        capacity=Int(segment_length)
    )  # aka: pvHLoad
    # Tracks scores for alignments that end with gaps in the query seq (horizontal gaps in visualization)
    var pv_e = List[SIMD[DType.uint8, SIMD_U8_WIDTH]](
        capacity=Int(segment_length)
    )  # aka: pvE
    # Stores the max scores seen in each column for traceback
    var pv_h_max = List[SIMD[DType.uint8, SIMD_U8_WIDTH]](
        capacity=Int(segment_length)
    )  # aka: pvHmax

    # init all of the above to zeros
    for _ in range(0, segment_length):
        pv_h_store.append(zero)
        pv_h_load.append(zero)
        pv_e.append(zero)
        pv_h_max.append(zero)

    var v_gap_open = SIMD[DType.uint8, SIMD_U8_WIDTH](
        gap_open_penalty
    )  # aka: vGap0
    var v_gap_ext = SIMD[DType.uint8, SIMD_U8_WIDTH](
        gap_extension_penalty
    )  # aka: vGapE
    var v_bias = SIMD[DType.uint8, SIMD_U8_WIDTH](bias)  # aka: vBias
    # Trace the highest scro of the whole SW matrix
    var v_max_score = zero  # aka: vMaxScore
    # Trace the highest score till the previous column
    var v_max_mark = zero  # aka: vMaxMark

    var edge: Int32 = 0
    var begin: Int32 = 0
    var end: Int32 = len(reference)
    var step: Int32 = 1

    # Outer loop to process the reference sequence
    if reference_direction == ReferenceDirection.Reverse:
        begin = len(reference) - 1
        end = -1
        step = -1

    var i = begin
    while likely(i != end):
        print("Outer loop:", i, "-", chr(Int(NUM_TO_NT[reference[i]])))
        # Initialize to 0, any errors in vH will be corrected in lazy_f
        var e = zero
        # Represents scores for alignments that end with gaps in the reference seq
        var v_f = zero  # aka: vF
        # the max score in the current column
        var v_max_column = zero  # aka: vMaxColumn
        # The score value currently being calculated
        var v_h = pv_h_store[segment_length - 1]  # aka: vH
        v_h = v_h.shift_right[1]()

        # Select the right vector from the query profile
        var profile_idx = reference[i].cast[DType.int32]() * segment_length

        # Swap the two score buffers
        swap(pv_h_load, pv_h_store)
        print("vH State:", v_h)
        print("pvHLoad State:", end="")
        for i in range(0, segment_length):
            print(pv_h_load[i], ", ", end="")
        print()

        print("pvHStore State:", end="")
        for i in range(0, segment_length):
            print(pv_h_store[i], ", ", end="")
        print()

        print("pvE State:", end="")
        for i in range(0, segment_length):
            print(pv_e[i], ", ", end="")
        print()

        print("pvHMax State:", end="")
        for i in range(0, segment_length):
            print(pv_h_max[i], ", ", end="")
        print()

        print("max_column State:", end="")
        for i in range(0, len(reference)):
            print(max_column[i], ", ", end="")
        print()

        print("end_read_column State:", end="")
        for i in range(0, len(reference)):
            print(end_query_column[i], ", ", end="")
        print()

        # Inner loop to process the query sequence
        for j in range(0, segment_length):
            print("\tInner loop for query sequence, checcking segment:", j)
            print("\tSegement J's query profile:", profile[profile_idx + j])
            # Add profile score to
            v_h = saturating_add(v_h, profile[profile_idx + j])
            v_h = saturating_sub(v_h, v_bias)  # adjust for bias
            print(
                "\t\tState of vH after adding profile and subtracting bias: ",
                v_h,
            )

            # Get max from current_cell_score, horizontal gap, and vertical gap score
            print("\t\t", "Get max from current cell / hgap / vgap")
            e = pv_e[j]
            v_h = max(v_h, e)
            v_h = max(v_h, v_f)
            v_max_column = max(v_max_column, v_h)
            print("\t\te State: ", e)
            print("\t\tvMaxColumn State: ", v_max_column)
            print("\t\tvH State after getting max: ", v_h)

            # Save current_cell_score
            pv_h_store[j] = v_h

            # update vE
            v_h = saturating_sub(v_h, v_gap_open)
            e = saturating_sub(e, v_gap_ext)
            e = max(e, v_h)
            pv_e[j] = e
            print("\t\te State after update: ", e)

            # update vF
            v_f = saturating_sub(v_f, v_gap_ext)
            v_f = max(v_f, v_h)
            print("\t\tvF State after update:", v_f)

            # load the next vH
            v_h = pv_h_load[j]
            print("\t\tnext vH:", v_h)

        print("\tStarting LazyF")
        # Lazy_F loop, disallows adjacent insertion and then deletion from SWPS3
        var break_out = False
        for k in range(0, SIMD_U8_WIDTH):
            v_f = v_f.shift_right[1]()
            print("\tLeft Shift vF:", v_f)
            print("\tWalking Segments")
            for j in range(0, segment_length):
                v_h = pv_h_store[j]
                v_h = max(v_h, v_f)
                print("\t\tvH after left shift and max vF", v_h)
                v_max_column = max(v_max_column, v_h)
                print("\t\t vMaxColumn State:", v_max_column)
                pv_h_store[j] = v_h

                v_h = saturating_sub(v_h, v_gap_open)
                v_f = saturating_sub(v_f, v_gap_ext)
                print("\t\tnew vH: ", v_h)
                print("\t\tnew vF: ", v_f)

                # Early termination check
                var v_temp = saturating_sub(v_f, v_h)
                var packed = v_temp == zero
                if unlikely(packed.reduce_and()):
                    print("\t\tCan terminate early")
                    break_out = True
                    break
            if break_out:
                break
        print("\t Done with main loops")

        # Check for new max score
        v_max_score = max(v_max_score, v_max_column)
        var equal_vector = v_max_mark == v_max_score
        if not equal_vector.reduce_and():
            # find max score in vector
            var temp = v_max_score.reduce_max()
            v_max_mark = v_max_score

            if likely(temp > max_score):
                max_score = temp
                if (max_score + bias) >= UInt8.MAX:
                    print("OVERFLOW")
                    break
                end_reference = i

                # Store the column with the highest alignment score in order to trace teh alignment ending position on the query
                for j in range(0, segment_length):
                    pv_h_max[j] = pv_h_store[j]

        # Record the max score of current column
        max_column[i] = v_max_column.reduce_max()
        if max_column[i] == terminate:
            # TODO: What's this doing?
            print("terminate early")
            break

        # Increment the while loop
        i += step

    # Trace the alignment ending position on query
    # var column_len = segment_length * SIMD_U8_WIDTH
    for i in range(0, segment_length):
        for j in range(0, SIMD_U8_WIDTH):
            if pv_h_max[i][j] == max_score:
                var temp = i + j * SIMD_U8_WIDTH
                if temp < Int(end_query):
                    end_query = temp

        # if pv_h_max[i / SIMD_U8_WIDTH][i % SIMD_U8_WIDTH] == max_score:
        #     var temp = i // SIMD_U8_WIDTH + (i % 16) * segment_length
        #     if temp < end_query:
        #         end_query = temp

    print("pvHMax State:", end="")
    for i in range(0, segment_length):
        print(pv_h_max[i], ", ", end="")
    print()

    print("max_column State:", end="")
    for i in range(0, len(reference)):
        print(max_column[i], ", ", end="")
    print()

    print("end_read_column State:", end="")
    for i in range(0, len(reference)):
        print(end_query_column[i], ", ", end="")
    print()
    # Find the most possible 2nd alignment
    var score_0 = max_score.cast[DType.uint16]() + bias.cast[
        DType.uint16
    ]() if max_score.cast[DType.uint16]() + bias.cast[
        DType.uint16
    ]() >= 255 else max_score.cast[
        DType.uint16
    ]()
    var bests = List[AlignmentEnd](
        AlignmentEnd(score_0, end_reference, end_query), AlignmentEnd(0, 0, 0)
    )

    edge = (end_reference - mask_length.cast[DType.int32]()) if (
        end_reference - mask_length.cast[DType.int32]()
    ) > 0 else 0
    for i in range(0, edge):
        if max_column[i].cast[DType.uint16]() > bests[1].score:
            bests[1].score = max_column[i].cast[DType.uint16]()
            bests[1].reference = i

    edge = (
        len(reference) if (end_reference + mask_length.cast[DType.int32]())
        > len(reference) else end_reference + mask_length.cast[DType.int32]()
    )
    for i in range(edge + 1, len(reference)):
        if max_column[i].cast[DType.uint16]() > bests[1].score:
            bests[1].score = max_column[i].cast[DType.uint16]()
            bests[1].reference = i

    return bests
