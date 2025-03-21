"""Striped Smith-Waterman Alignment."""

from builtin.math import max
from collections import InlineArray
from math import sqrt
from memory import pack_bits, memset_zero

from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix


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


@value
struct Cigar:
    var seq: List[UInt32]


@value
@register_passable
struct ReferenceDirection:
    """Direction of the reference sequence."""

    var value: UInt8
    alias Forward = Self(0)
    alias Reverse = Self(1)

    fn __eq__(read self, read other: Self) -> Bool:
        return self.value == other.value


@value
@register_passable
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

    fn __str__(read self) -> String:
        if self.value == Self.Byte.value:
            return "byte"
        elif self.value == self.Word.value:
            return "word"
        else:
            return "adaptive"


@value
@register_passable("trivial")
struct Alignment:
    var score1: UInt16
    var ref_begin1: Int32
    var ref_end1: Int32
    var read_begin1: Int32
    var read_end1: Int32


@value
@register_passable("trivial")
struct AlignmentResult:
    var best: AlignmentEnd


@value
@register_passable("trivial")
struct AlignmentEnd:
    var score: UInt16
    var reference: Int32  # 0-based
    var query: Int32  # alignment ending position on query, 0-based


@value
struct ProfileVectors[dt: DType, width: Int]:
    var pv_h_store: List[SIMD[dt, width]]
    var pv_h_load: List[SIMD[dt, width]]
    var pv_e: List[SIMD[dt, width]]
    var pv_h_max: List[SIMD[dt, width]]
    var zero: SIMD[dt, width]
    var max_column: List[SIMD[dt, 1]]
    var end_query_column: List[Int32]
    var segment_length: Int32
    var query_len: Int32

    fn __init__(out self, query_len: Int32):
        var segment_length = (query_len + width - 1) // width

        var zero = SIMD[dt, width](0)
        # Stores the H values (scores) for the current row of the dynamic programming matrix
        var pv_h_store = List[SIMD[dt, width]](
            capacity=Int(segment_length)
        )  # aka: pvHStore
        # Contains scores from the previous row that will be loaded for calculation
        var pv_h_load = List[SIMD[dt, width]](
            capacity=Int(segment_length)
        )  # aka: pvHLoad
        # Tracks scores for alignments that end with gaps in the query seq (horizontal gaps in visualization)
        var pv_e = List[SIMD[dt, width]](
            capacity=Int(segment_length)
        )  # aka: pvE
        # Stores the max scores seen in each column for traceback
        var pv_h_max = List[SIMD[dt, width]](
            capacity=Int(segment_length)
        )  # aka: pvHmax

        # init all of the above to zeros
        memset_zero(pv_h_store.unsafe_ptr(), Int(segment_length))
        memset_zero(pv_h_load.unsafe_ptr(), Int(segment_length))
        memset_zero(pv_e.unsafe_ptr(), Int(segment_length))
        memset_zero(pv_h_max.unsafe_ptr(), Int(segment_length))

        # List to record the largest score of each reference position
        var max_column = List[SIMD[dt, 1]]()
        # List to record the alignment query ending position of the largest score of each reference position
        var end_query_column = List[Int32]()

        self.pv_h_store = pv_h_store
        self.pv_h_load = pv_h_load
        self.pv_e = pv_e
        self.pv_h_max = pv_h_max
        self.zero = zero
        self.segment_length = segment_length
        self.query_len = query_len
        self.max_column = max_column
        self.end_query_column = end_query_column

    fn zero_out(mut self):
        memset_zero(self.pv_h_store.unsafe_ptr(), Int(self.segment_length))
        memset_zero(self.pv_h_load.unsafe_ptr(), Int(self.segment_length))
        memset_zero(self.pv_e.unsafe_ptr(), Int(self.segment_length))
        memset_zero(self.pv_h_max.unsafe_ptr(), Int(self.segment_length))

    fn init_columns(mut self, ref_len: Int):
        self.max_column.resize(ref_len, 0)
        self.end_query_column.resize(ref_len, 0)
        memset_zero(self.max_column.unsafe_ptr(), ref_len)
        memset_zero(self.end_query_column.unsafe_ptr(), ref_len)


@value
struct Profile[SIMD_U8_WIDTH: Int, SIMD_U16_WIDTH: Int]:
    alias ByteVProfile = List[SIMD[DType.uint8, SIMD_U8_WIDTH]]
    alias WordVProfile = List[SIMD[DType.uint16, SIMD_U16_WIDTH]]

    var profile_byte: Optional[Self.ByteVProfile]
    var profile_word: Optional[Self.WordVProfile]
    var byte_vectors: ProfileVectors[DType.uint8, SIMD_U8_WIDTH]
    var word_vectors: ProfileVectors[DType.uint16, SIMD_U16_WIDTH]
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
        # print("Intializing a profile for a query of len", len(query))
        var profile_byte: Optional[Self.ByteVProfile] = None
        var profile_word: Optional[Self.WordVProfile] = None
        var bias: UInt8 = 0
        if score_size == ScoreSize.Byte or score_size == ScoreSize.Adaptive:
            # print("Score size is byte")
            # print(
            #     len(score_matrix.values)
            # )  # Problem is around here, score matrix is garbage
            # Find the bias to use in the substitution matrix
            # The bias will be smallest value in the scoring matrix
            var bias_tmp: Int8 = 0
            for i in range(0, len(score_matrix.values)):
                if score_matrix.values[i] < bias_tmp:
                    bias_tmp = score_matrix.values[i]
            bias = abs(bias_tmp).cast[DType.uint8]()
            # print("Calling generate profile")
            profile_byte = Self.generate_query_profile[
                DType.uint8, SIMD_U8_WIDTH
            ](query, score_matrix, bias)

        if score_size == ScoreSize.Word or score_size == ScoreSize.Adaptive:
            # print("Score size is adaptive")
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
            ProfileVectors[DType.uint8, SIMD_U8_WIDTH](len(query)),
            ProfileVectors[DType.uint16, SIMD_U16_WIDTH](len(query)),
            len(query),
            score_matrix.size,
            bias,
        )

    @staticmethod
    fn generate_query_profile[
        T: DType, size: Int
    ](
        query: Span[UInt8], read score_matrix: ScoringMatrix, bias: UInt8
    ) -> List[SIMD[T, size]]:
        """Divide the query into segments."""
        # print("Generating query profile")
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


fn ssw_align[
    SIMD_U8_WIDTH: Int, SIMD_U16_WIDTH: Int
](
    mut profile: Profile[SIMD_U8_WIDTH, SIMD_U16_WIDTH],
    read matrix: ScoringMatrix,
    reference: Span[UInt8],
    query: Span[UInt8],
    mut reverse_profile: Profile[SIMD_U8_WIDTH, SIMD_U16_WIDTH],
    *,
    gap_open_penalty: UInt8 = 3,
    gap_extension_penalty: UInt8 = 1,
    return_only_alignment_end: Bool = False,
    mask_length: Int32 = 15,  # for second best score
) -> Optional[Alignment]:
    # Find the alignment scores and ending positions
    var bests: AlignmentResult

    var used_word = False

    if profile.profile_byte:
        # print("Doing Byte SW")
        bests = sw[DType.uint8, SIMD_U8_WIDTH](
            reference,
            ReferenceDirection.Forward,
            profile.query_len,
            gap_open_penalty,
            gap_extension_penalty,
            profile.profile_byte.value(),
            profile.byte_vectors,
            -1,
            profile.bias,
            mask_length,
        )
        if profile.profile_word and bests.best.score == 255:
            bests = sw[DType.uint16, SIMD_U16_WIDTH](
                reference,
                ReferenceDirection.Forward,
                profile.query_len,
                gap_open_penalty.cast[DType.uint16](),
                gap_extension_penalty.cast[DType.uint16](),
                profile.profile_word.value(),
                profile.word_vectors,
                -1,
                profile.bias.cast[DType.uint16](),
                mask_length,
            )
            bests.best.score -= profile.bias.cast[DType.uint16]()
            used_word = True
        elif bests.best.score == 255:
            print(
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
            profile.profile_word.value(),
            profile.word_vectors,
            -1,
            profile.bias.cast[DType.uint16](),
            mask_length,
        )
        bests.best.score -= profile.bias.cast[DType.uint16]()

        used_word = True
    else:
        print("Failed to provide a valid query profile")
        return None

    if bests.best.score <= 0:
        return None

    var score1 = bests.best.score
    var ref_end1 = bests.best.reference
    var read_end1 = bests.best.query

    var score2: UInt16 = 0
    var ref_end2: Int32 = -1

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
            reference,
            ReferenceDirection.Reverse,
            reverse_profile.query_len,
            gap_open_penalty,
            gap_extension_penalty,
            reverse_profile.profile_byte.value(),
            reverse_profile.byte_vectors,
            -1,
            reverse_profile.bias,
            mask_length,
        )
    else:
        bests_rev = sw[DType.uint16, SIMD_U16_WIDTH](
            reference,
            ReferenceDirection.Reverse,
            reverse_profile.query_len,
            gap_open_penalty.cast[DType.uint16](),
            gap_extension_penalty.cast[DType.uint16](),
            reverse_profile.profile_word.value(),
            reverse_profile.word_vectors,
            -1,
            reverse_profile.bias.cast[DType.uint16](),
            mask_length,
        )
    var ref_begin1 = bests_rev.best.reference
    var read_begin1 = read_end1 - bests_rev.best.query
    # if fwd score  > rev score, we might be missing something?

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
    mut p_vecs: ProfileVectors[dt, width],
    terminate: SIMD[dt, 1],
    bias: SIMD[dt, 1],
    mask_length: Int32,
) -> AlignmentResult:
    """Smith-Waterman local alignment.

    Arguments:
        terminate: The best alignment score, used to terminate the matrix calc when locating the alignment beginning point. If this score is set to 0, it will not be used.

    """
    p_vecs.zero_out()
    p_vecs.init_columns(len(reference))
    var max_score = UInt8(0).cast[dt]()
    var end_query: Int32 = query_len - 1
    var end_reference: Int32 = -1  # 0 based best alignment ending point; initialized as isn't aligned -1
    # var segment_length = (query_len + width - 1) // width
    var segment_length = p_vecs.segment_length

    # Note:
    # H - Score for match / mismatch (diagonal move)
    # E - Score for gap in query (horizontal move)
    # F - Score for gap in reference (vertical move)
    # print("Segment length:", segment_length)
    # print("RefLen: ", len(reference))
    # print("querylen: ", query_len)
    # print("SIMD DType", dt)
    # print("SIMD WIDTH", width)
    # print("bias", bias)
    # for i in range(
    #     0, ScoringMatrix.blosm50().size
    # ):  # TODO: hardcoded, should be length alphabet
    #     print(chr(Int(NUM_TO_AA[i])), ": ", sep="", end="")
    #     for j in range(0, segment_length):
    #         print(profile[i * segment_length + j], ", ", end="")
    #     print()
    # print("Initializing")

    # List to record the largest score of each reference position
    # var max_column = List[SIMD[dt, 1]](capacity=len(reference))
    # List to record the alignment query ending position of the largest score of each reference position
    # var end_query_column = List[Int32](capacity=len(reference))

    # for _ in range(0, len(reference)):
    #     p_vecs.max_column.append(UInt8(0).cast[dt]())
    #     p_vecs.end_query_column.append(0)

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
    # print("Done with init")
    var i = begin
    while i != end:
        # print(
        #     "Outer loop:", i, "-"
        # )  # , chr(Int(NUM_TO_AA[Int(reference[i])])))
        # Initialize to 0, any errors in vH will be corrected in lazy_f
        var e = zero
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
        # print("vH State:", v_h)
        # print("pvHLoad State:", end="")
        # for i in range(0, segment_length):
        #     print(p_vecs.pv_h_load[i], ", ", end="")
        # print()

        # print("pvHStore State:", end="")
        # for i in range(0, segment_length):
        #     print(p_vecs.pv_h_store[i], ", ", end="")
        # print()

        # print("pvE State:", end="")
        # for i in range(0, segment_length):
        #     print(p_vecs.pv_e[i], ", ", end="")
        # print()

        # print("pvHMax State:", end="")
        # for i in range(0, segment_length):
        #     print(p_vecs.pv_h_max[i], ", ", end="")
        # print()

        # print("max_column State:", end="")
        # for i in range(0, len(reference)):
        #     print(p_vecs.max_column[i], ", ", end="")
        # print()

        # print("end_read_column State:", end="")
        # for i in range(0, len(reference)):
        #     print(p_vecs.end_query_column[i], ", ", end="")
        # print()

        # Inner loop to process the query sequence
        for j in range(0, segment_length):
            # print("\tInner loop for query sequence, checcking segment:", j)
            # print("\tSegement J's query profile:", profile[profile_idx + j])
            # Add profile score to
            v_h = saturating_add(v_h, profile[profile_idx + j])
            v_h = saturating_sub(v_h, v_bias)  # adjust for bias
            # print(
            #     "\t\tState of vH after adding profile and subtracting bias: ",
            #     v_h,
            # )

            # Get max from current_cell_score, horizontal gap, and vertical gap score
            # print("\t\t", "Get max from current cell / hgap / vgap")
            e = p_vecs.pv_e[j]
            v_h = max(v_h, e)
            v_h = max(v_h, v_f)
            v_max_column = max(v_max_column, v_h)
            # print("\t\te State: ", e)
            # print("\t\tvMaxColumn State: ", v_max_column)
            # print("\t\tvH State after getting max: ", v_h)

            # Save current_cell_score
            p_vecs.pv_h_store[j] = v_h

            # update vE
            v_h = saturating_sub(v_h, v_gap_open)
            e = saturating_sub(e, v_gap_ext)
            e = max(e, v_h)
            p_vecs.pv_e[j] = e
            # print("\t\te State after update: ", e)

            # update vF
            v_f = saturating_sub(v_f, v_gap_ext)
            v_f = max(v_f, v_h)
            # print("\t\tvF State after update:", v_f)

            # load the next vH
            v_h = p_vecs.pv_h_load[j]
            # print("\t\tnext vH:", v_h)

        # print("\tStarting LazyF")
        # Lazy_F loop, disallows adjacent insertion and then deletion from SWPS3
        # Possible speedup - check if v_f has any updates to start with
        # var break_out = (v_f == zero).reduce_and()
        # var k = 0
        # while not break_out and k < width:
        # k += 1
        var break_out = False

        @parameter
        for _k in range(0, width):
            v_f = v_f.shift_right[1]()
            # print("\tLeft Shift vF:", v_f)
            # print("\tWalking Segments")
            for j in range(0, segment_length):
                # print("\t\tvF:               ", v_f)
                v_h = p_vecs.pv_h_store[j]
                # print("\t\tvH from store:    ", v_h)
                v_h = max(v_h, v_f)
                # print("\t\tvH post max store:", v_h)
                # print("\t\tvH after left shift and max vF", v_h)
                v_max_column = max(v_max_column, v_h)
                # print("\t\t vMaxColumn State:", v_max_column)
                p_vecs.pv_h_store[j] = v_h

                v_h = saturating_sub(v_h, v_gap_open)
                v_f = saturating_sub(v_f, v_gap_ext)
                # print("\t\tnew vH: ", v_h)
                # print("\t\tnew vF: ", v_f)

                # Early termination check
                var v_temp = saturating_sub(v_f, v_h)
                # print("\t\tnew v_temp: ", v_temp)
                var packed = v_temp == zero
                if packed.reduce_and():
                    # print("\t\tCan terminate early")
                    break_out = True
                    break
            if break_out:
                break
        # print("\t Done with main loops")

        # Check for new max score
        v_max_score = max(v_max_score, v_max_column)
        var equal_vector = v_max_mark == v_max_score
        if not equal_vector.reduce_and():
            # find max score in vector
            var temp = v_max_score.reduce_max()
            v_max_mark = v_max_score

            if temp > max_score:
                max_score = temp
                if (max_score + bias) >= SIMD[dt, 1].MAX:
                    # print("OVERFLOW")
                    break
                end_reference = i

                # Store the column with the highest alignment score in order to trace teh alignment ending position on the query
                for j in range(0, segment_length):
                    p_vecs.pv_h_max[j] = p_vecs.pv_h_store[j]

        # Record the max score of current column
        p_vecs.max_column[i] = v_max_column.reduce_max()
        if p_vecs.max_column[i] == terminate:
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

    # print("pvHMax State:", end="")
    # for i in range(0, segment_length):
    #     print(p_vecs.pv_h_max[i], ", ", end="")
    # print()

    # print("max_column State:", end="")
    # for i in range(0, len(reference)):
    #     print(p_vecs.max_column[i], ", ", end="")
    # print()

    # print("end_read_column State:", end="")
    # for i in range(0, len(reference)):
    #     print(p_vecs.end_query_column[i], ", ", end="")
    # print()
    # Find the most possible 2nd alignment
    var score_0 = max_score + bias if max_score + bias >= 255 else max_score
    var bests = AlignmentResult(
        AlignmentEnd(score_0.cast[DType.uint16](), end_reference, end_query),
    )

    # Candidate for SIMD?
    # edge = (end_reference - mask_length) if (
    #     end_reference - mask_length
    # ) > 0 else 0
    # for i in range(0, edge):
    #     if p_vecs.max_column[i] > bests.second_best.score.cast[dt]():
    #         bests.second_best.score = p_vecs.max_column[i].cast[DType.uint16]()
    #         bests.second_best.reference = i

    # edge = (
    #     len(reference) if (end_reference + mask_length)
    #     > len(reference) else end_reference + mask_length
    # )
    # for i in range(edge + 1, len(reference)):
    #     if p_vecs.max_column[i] > bests.second_best.score.cast[dt]():
    #         bests.second_best.score = p_vecs.max_column[i].cast[DType.uint16]()
    #         bests.second_best.reference = i

    # print(
    #     "BEST: score:",
    #     bests.best.score,
    #     "ref:",
    #     bests.best.reference,
    #     "read:",
    #     bests.best.query,
    # )
    return bests
