from collections import Optional
from collections.string import StringSlice
from memory import Span, AddressSpace, UnsafePointer
from sys.info import alignof

from ishlib.gpu.dynamic_2d_matrix import Dynamic2DMatrix, StorageFormat
from ishlib.matcher.alignment import AlignmentResult, TargetSpan
from ishlib.matcher.alignment.scoring_matrix import (
    ScoringMatrix,
    BasicScoringMatrix,
)


@value
@register_passable("trivial")
struct SGResult(StringableRaising):
    var query: Int
    """0-based inclusive index in the query."""
    var target: Int
    """0-based inclusive index in the target."""
    var score: Int32
    """Alignment score."""

    fn __str__(read self) raises -> String:
        return String("query={}, target={}, score={}").format(
            self.query, self.target, self.score
        )


fn semi_global_parasail_start_end[
    DT: DType = DType.int32
](
    query: Span[UInt8],
    target: Span[UInt8],
    rev_query: Span[UInt8],
    rev_target: Span[UInt8],
    read scoring_matrix: ScoringMatrix,
    *,
    gap_open_penalty: Scalar[DT] = -3,
    gap_extension_penalty: Scalar[DT] = -1,
    free_query_start_gaps: Bool = False,
    free_query_end_gaps: Bool = False,
    free_target_start_gaps: Bool = False,
    free_target_end_gaps: Bool = False,
    score_cutoff: Int = 1,
) -> Optional[AlignmentResult]:
    """Semi-global alignment to find the start end end points of the query on the target.

    Arguments:
        query: scoring_matrix encoded query sequence.
        target: scoring_matrix encoded target sequence.
        scoring_matrix: ScoringMatrix for match/mismatch
        gap_open_penalty: The cost (negative) of opening a gap.
        gap_extension_penalty: The cost (negative) of extending an existing gap.
        free_query_start_gaps: Gaps at the start of the query are free.
        free_query_end_gaps: Gaps at the end of the query are free.
        free_target_start_gaps: Gaps at the start of the target are free.
        free_target_end_gaps: Gaps at the end of the target are free.
    """
    var ends = semi_global_parasail[DT=DT](
        query,
        target,
        scoring_matrix,
        gap_open_penalty=gap_open_penalty,
        gap_extension_penalty=gap_extension_penalty,
        free_query_start_gaps=free_query_start_gaps,
        free_query_end_gaps=free_query_end_gaps,
        free_target_start_gaps=free_target_start_gaps,
        free_target_end_gaps=free_target_end_gaps,
    )

    if ends.score <= score_cutoff:
        return None

    var starts = semi_global_parasail[DT=DT](
        rev_query,
        rev_target,
        scoring_matrix,
        gap_open_penalty=gap_open_penalty,
        gap_extension_penalty=gap_extension_penalty,
        # NOTE: the flipped free gap bools
        free_query_start_gaps=free_query_end_gaps,
        free_query_end_gaps=free_query_start_gaps,
        free_target_start_gaps=free_target_end_gaps,
        free_target_end_gaps=free_target_start_gaps,
    )
    return AlignmentResult(
        ends.score,
        None,
        None,
        TargetSpan(len(target) - starts.target - 1, ends.target),
    )


fn semi_global_parasail[
    DT: DType = DType.int32
](
    query: Span[UInt8],
    target: Span[UInt8],
    read scoring_matrix: ScoringMatrix,
    *,
    gap_open_penalty: Scalar[DT] = -3,
    gap_extension_penalty: Scalar[DT] = -1,
    free_query_start_gaps: Bool = False,
    free_query_end_gaps: Bool = False,
    free_target_start_gaps: Bool = False,
    free_target_end_gaps: Bool = False,
) -> SGResult:
    """Semi-global alignment to find the end points of the query and target.

    Arguments:
        query: scoring_matrix encoded query sequence.
        target: scoring_matrix encoded target sequence.
        scoring_matrix: ScoringMatrix for match/mismatch
        gap_open_penalty: The cost (negative) of opening a gap.
        gap_extension_penalty: The cost (negative) of extending an existing gap.
        free_query_start_gaps: Gaps at the start of the query are free.
        free_query_end_gaps: Gaps at the end of the query are free.
        free_target_start_gaps: Gaps at the start of the target are free.
        free_target_end_gaps: Gaps at the end of the target are free.
    """
    alias NUM = Scalar[DT]

    if len(query) == 0 or len(target) == 0:
        return SGResult(0, 0, 0)

    # Init the sizes
    var cols = len(query) + 1  # s2
    var rows = len(target) + 1  # s1

    var H = List[NUM](capacity=cols)
    var F = List[NUM](capacity=cols)
    for _ in range(0, cols):
        H.append(0)
        F.append(0)

    # Upper Left corner
    H[0] = 0
    F[0] = NUM.MIN // 2

    # Key for parasail comp, but note that the way it passes things in via CLI isn't a 1:1 map with this
    # s1 == rows == target
    # s2 == cols == query

    var score = NUM.MIN // 2
    var end_query = len(query)
    var end_target = len(target)

    # First Row
    if free_query_start_gaps:
        for j in range(1, cols):
            H[j] = 0
            F[j] = NUM.MIN // 2
    else:
        for j in range(1, cols):
            H[j] = gap_open_penalty + ((j - 1) * gap_extension_penalty)
            F[j] = NUM.MIN // 2

    # Iterate over the target sequence
    for i in range(1, rows - 1):
        # Init first column
        var NH = H[0]
        var WH = 0 if free_target_start_gaps else gap_open_penalty + (
            (i - 1) * gap_extension_penalty
        )
        var E = NUM.MIN // 2
        H[0] = WH

        for j in range(1, cols):
            var NWH = NH
            NH = H[j]
            var F_open = NH + gap_open_penalty
            var F_ext = F[j] + gap_extension_penalty
            F[j] = max(F_open, F_ext)
            var E_open = WH + gap_open_penalty
            var E_ext = E + gap_extension_penalty
            E = max(E_open, E_ext)
            var H_dag = NWH + scoring_matrix.get(
                Int(target[i - 1]), Int(query[j - 1])
            ).cast[DT]()
            WH = max(max(H_dag, E), F[j])
            H[j] = WH

        if free_target_end_gaps and WH > score:
            score = WH
            end_target = i - 1
            end_query = len(query) - 1

    # handle final row specially
    # i == target length
    var NH = H[0]
    var WH = 0 if free_target_start_gaps else gap_open_penalty + (
        (len(target) - 1) * gap_extension_penalty
    )
    var E = NUM.MIN // 2
    H[0] = WH
    for j in range(1, cols):
        var NWH = NH
        NH = H[j]
        var F_open = NH + gap_open_penalty
        var F_ext = F[j] + gap_extension_penalty
        F[j] = max(F_open, F_ext)
        var E_open = WH + gap_open_penalty
        var E_ext = E + gap_extension_penalty
        E = max(E_open, E_ext)
        var H_dag = (
            NWH
            + scoring_matrix.get(
                Int(target[len(target) - 1]), Int(query[j - 1])
            ).cast[DT]()
        )
        WH = max(max(H_dag, E), F[j])
        H[j] = WH

        if free_target_end_gaps and free_query_end_gaps:
            if WH > score:
                score = WH
                end_target = len(target) - 1
                end_query = j - 1
            elif WH == score and j - 1 < end_query:
                end_target = len(target) - 1
                end_query = j - 1

        elif free_query_end_gaps:
            if WH > score:
                score = WH
                end_target = len(target) - 1
                end_query = j - 1

    if (free_target_end_gaps and WH > score) or (
        not free_target_end_gaps and not free_query_end_gaps
    ):
        score = WH
        end_target = len(target) - 1
        end_query = len(query) - 1

    return SGResult(end_query, end_target, score.cast[DType.int32]())


fn semi_global_parasail_gpu[
    DT: DType = DType.int32,
    *,
    max_query_length: UInt,
    address_space: AddressSpace = AddressSpace(0),
    target_address_space: AddressSpace = AddressSpace(0),
    matrix_skip_lookup: Bool = False,
](
    query_ptr: UnsafePointer[UInt8, address_space=address_space],
    query_len: Int,
    target_tensor: UnsafePointer[UInt8, address_space=target_address_space],
    target_rows: UInt,  # equivalent to max_target_len
    target_cols: UInt,
    target_of_interest_column: UInt,
    target_len: UInt,
    scoring_matrix: BasicScoringMatrix[
        address_space=address_space, no_lookup=matrix_skip_lookup
    ],
    gap_open_penalty: Scalar[DT] = -3,
    gap_extension_penalty: Scalar[DT] = -1,
    free_query_start_gaps: Bool = False,
    free_query_end_gaps: Bool = False,
    free_target_start_gaps: Bool = False,
    free_target_end_gaps: Bool = False,
) -> SGResult:
    """Semi-global alignment to find the end points of the query and target.

    Note, this uses the opposite ordering as parasail, which can result in differnt.
    To compare against parasail make sure to swap the query and target, and then also
    track the swap in the output file.

    Arguments:
        query_ptr: scoring_matrix encoded query sequence pointer.
        query_len: The length of the query sequence
        target_tensor: scoring_matrix encoded target sequence pointer.
        target_rows: the number of rows in the targets 2D matrix.
        target_cols: the number of cols in the targets 2D matrix.
        target_of_interest_column: the column that has the target of interest.
        target_len: the length of the array backing the target_tensor.
        scoring_matrix: ScoringMatrix for match/mismatch
        gap_open_penalty: The cost (negative) of opening a gap.
        gap_extension_penalty: The cost (negative) of extending an existing gap.

    Paramters
        free_query_start_gaps: Gaps at the start of the query are free.
        free_query_end_gaps: Gaps at the end of the query are free.
        free_target_start_gaps: Gaps at the start of the target are free.
        free_target_end_gaps: Gaps at the end of the target are free.
    """

    alias NUM = Scalar[DT]
    var HALF_MIN = NUM.MIN // 2
    # var scoring_matrix = BadScoring()

    var coords = Dynamic2DMatrix[storage_format = StorageFormat.ColumnMajor](
        target_rows, target_cols
    )

    if query_len == 0 or target_len == 0:
        return SGResult(0, 0, 0)

    # Init the sizes
    var cols = query_len + 1  # s2
    var rows = target_len + 1  # s12

    var H = InlineArray[NUM, max_query_length](fill=0)
    var F = InlineArray[NUM, max_query_length](fill=HALF_MIN)

    var score = HALF_MIN
    var end_query = query_len
    var end_target = target_len

    # First Row
    # @parameter
    if not free_query_start_gaps:
        for j in range(1, cols):
            H[j] = gap_open_penalty + ((j - 1) * gap_extension_penalty)

    var NH: Scalar[DT]
    var WH: Scalar[DT]

    # Iterate over the target sequence
    for i in range(1, rows - 1):
        # Init first column
        NH = H[0]
        WH = NUM(0)

        # @parameter
        if not free_target_start_gaps:
            WH = gap_open_penalty + ((i - 1) * gap_extension_penalty)

        var E = HALF_MIN
        H[0] = WH

        for j in range(1, cols):
            var NWH = NH
            NH = H[j]
            var F_open = NH + gap_open_penalty
            var F_ext = F[j] + gap_extension_penalty
            F[j] = max(F_open, F_ext)
            var E_open = WH + gap_open_penalty
            var E_ext = E + gap_extension_penalty
            E = max(E_open, E_ext)
            var H_dag = NWH + scoring_matrix.get(
                Int(
                    target_tensor[
                        coords.cord2idx(i - 1, target_of_interest_column)
                    ]
                ),
                Int(query_ptr[j - 1]),
            ).cast[DT]()
            WH = max(max(H_dag, E), F[j])
            H[j] = WH

        # @parameter
        if free_target_end_gaps:
            if WH > score:
                score = WH
                end_target = i - 1
                end_query = query_len - 1

    # handle final row specially
    NH = H[0]
    WH = NUM(0)

    # @parameter
    if not free_target_start_gaps:
        WH = gap_open_penalty + ((target_len - 1) * gap_extension_penalty)
    var E = HALF_MIN
    H[0] = WH
    for j in range(1, cols):
        var NWH = NH
        NH = H[j]
        var F_open = NH + gap_open_penalty
        var F_ext = F[j] + gap_extension_penalty
        F[j] = max(F_open, F_ext)
        var E_open = WH + gap_open_penalty
        var E_ext = E + gap_extension_penalty
        E = max(E_open, E_ext)
        var H_dag = NWH + scoring_matrix.get(
            Int(
                target_tensor[
                    coords.cord2idx(target_len - 1, target_of_interest_column)
                ]
            ),
            Int(query_ptr[j - 1]),
        ).cast[DT]()
        WH = max(max(H_dag, E), F[j])
        H[j] = WH

        # @parameter
        if free_target_end_gaps and free_query_end_gaps:
            if WH > score:
                score = WH
                end_target = target_len - 1
                end_query = j - 1
            elif WH == score and j - 1 < end_query:
                end_target = target_len - 1
                end_query = j - 1
        elif free_query_end_gaps:
            if WH > score:
                score = WH
                end_target = target_len - 1
                end_query = j - 1

    # @parameter
    if free_target_end_gaps:
        if WH > score:
            score = WH
            end_target = target_len - 1
            end_query = query_len - 1

    elif not free_target_end_gaps and not free_query_end_gaps:
        score = WH
        end_target = target_len - 1
        end_query = query_len - 1

    return SGResult(end_query, end_target, score.cast[DType.int32]())


# This impl fixes the ordering so the query is in the outer loop,
# which makes outputs match default parasail. But this means that
# the memory used == length of target, which is not great.
# So we'll stick with the flipped ordering, which has been validated
# against parasail doing flipped ordering as well.
# fn semi_global_parasail_gpu_parasail[
#     DT: DType,
#     *,
#     max_target_length: UInt,
#     address_space: AddressSpace = AddressSpace(0),
#     target_address_space: AddressSpace = AddressSpace(0),
#     free_query_start_gaps: Bool = False,
#     free_query_end_gaps: Bool = False,
#     free_target_start_gaps: Bool = False,
#     free_target_end_gaps: Bool = False,
# ](
#     query_ptr: UnsafePointer[UInt8, address_space=address_space],
#     query_len: Int,
#     target_tensor: UnsafePointer[UInt8, address_space=target_address_space],
#     target_rows: UInt,
#     target_cols: UInt,
#     target_of_interest_column: UInt,
#     target_len: UInt,
#     scoring_matrix: BasicScoringMatrix[address_space=address_space],
#     gap_open_penalty: Scalar[DT] = -3,
#     gap_extension_penalty: Scalar[DT] = -1,
# ) -> SGResult:
#     """Semi-global alignment to find the end points of the query and target.

#     Arguments:
#         query_ptr: scoring_matrix encoded query sequence pointer.
#         query_len: The length of the query sequence
#         target_tensor: scoring_matrix encoded target sequence pointer.
#         target_rows: the number of rows in the targets 2D matrix.
#         target_cols: the number of cols in the targets 2D matrix.
#         target_of_interest_column: the column that has the target of interest.
#         target_len: the length of the array backing the target_tensor.
#         scoring_matrix: ScoringMatrix for match/mismatch
#         gap_open_penalty: The cost (negative) of opening a gap.
#         gap_extension_penalty: The cost (negative) of extending an existing gap.

#     Paramters
#         max_target_length: The maximum allowed length of a target. This is used to pre-allocate memory.
#         free_query_start_gaps: Gaps at the start of the query are free.
#         free_query_end_gaps: Gaps at the end of the query are free.
#         free_target_start_gaps: Gaps at the start of the target are free.
#         free_target_end_gaps: Gaps at the end of the target are free.
#     """

#     # In parasail:
#     # - s1 = query
#     # - s2 = target
#     #
#     # In this I've swapped s1 and s2... this is going to get confusing

#     alias NUM = Scalar[DT]
#     var HALF_MIN = NUM.MIN // 2
#     # var scoring_matrix = BadScoring()

#     var coords = Dynamic2DMatrix[storage_format = StorageFormat.ColumnMajor](
#         target_rows, target_cols
#     )

#     if query_len == 0 or target_len == 0:
#         return SGResult(0, 0, 0)

#     # Init the sizes
#     var rows = query_len + 1  # s2
#     var cols = target_len + 1  # s2

#     var H = InlineArray[NUM, max_target_length + 1](fill=0)
#     var F = InlineArray[NUM, max_target_length + 1](fill=HALF_MIN)

#     var score = HALF_MIN
#     var end_query = query_len
#     var end_target = target_len

#     # First Row
#     @parameter
#     if not free_target_start_gaps:
#         for j in range(1, cols):
#             H[j] = gap_open_penalty + ((j - 1) * gap_extension_penalty)

#     var NH = H[0]
#     var WH = NUM(0)

#     # Iterate over the query sequence
#     for i in range(1, rows - 1):
#         # Init first column
#         NH = H[0]
#         WH = NUM(0)

#         @parameter
#         if not free_query_start_gaps:
#             WH = gap_open_penalty + ((i - 1) * gap_extension_penalty)

#         var E = HALF_MIN
#         H[0] = WH

#         for j in range(1, cols):
#             var NWH = NH
#             NH = H[j]
#             var F_open = NH + gap_open_penalty
#             var F_ext = F[j] + gap_extension_penalty
#             F[j] = max(F_open, F_ext)
#             var E_open = WH + gap_open_penalty
#             var E_ext = E + gap_extension_penalty
#             E = max(E_open, E_ext)
#             var H_dag = NWH + scoring_matrix.get(
#                 Int(query_ptr[i - 1]),
#                 Int(
#                     target_tensor[
#                         coords.cord2idx(j - 1, target_of_interest_column)
#                     ]
#                 ),
#             ).cast[DT]()
#             WH = max(max(H_dag, E), F[j])
#             H[j] = WH

#         @parameter
#         if free_query_end_gaps:
#             if WH > score:
#                 score = WH
#                 end_query = i - 1
#                 end_target = target_len - 1

#     # handle final row specially
#     NH = H[0]
#     WH = NUM(0)

#     @parameter
#     if not free_query_start_gaps:
#         WH = gap_open_penalty + ((target_len - 1) * gap_extension_penalty)
#     var E = HALF_MIN
#     H[0] = WH
#     for j in range(1, cols):
#         var NWH = NH
#         NH = H[j]
#         var F_open = NH + gap_open_penalty
#         var F_ext = F[j] + gap_extension_penalty
#         F[j] = max(F_open, F_ext)
#         var E_open = WH + gap_open_penalty
#         var E_ext = E + gap_extension_penalty
#         E = max(E_open, E_ext)
#         var H_dag = NWH + scoring_matrix.get(
#             Int(query_ptr[query_len - 1]),
#             Int(
#                 target_tensor[coords.cord2idx(j - 1, target_of_interest_column)]
#             ),
#         ).cast[DT]()
#         WH = max(max(H_dag, E), F[j])
#         H[j] = WH

#         @parameter
#         if free_target_end_gaps and free_query_end_gaps:
#             if WH > score:
#                 score = WH
#                 end_target = j - 1
#                 end_query = query_len - 1
#             elif WH == score and j - 1 < end_target:
#                 end_target = j - 1
#                 end_query = query_len - 1
#         elif free_target_end_gaps:
#             if WH > score:
#                 score = WH
#                 end_target = j - 1
#                 end_query = query_len - 1

#     @parameter
#     if free_query_end_gaps:
#         if WH > score:
#             score = WH
#             end_target = target_len - 1
#             end_query = query_len - 1

#     elif not free_target_end_gaps and not free_query_end_gaps:
#         score = WH
#         end_target = target_len - 1
#         end_query = query_len - 1

#     return SGResult(end_query, end_target, score.cast[DType.int32]())
