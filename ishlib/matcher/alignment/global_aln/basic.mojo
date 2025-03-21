"""Globally align any two sequences."""
from collections import Optional
from collections.string import StringSlice
from memory import Span

from ishlib.matcher.alignment import AlignmentResult
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix


fn needleman_wunsch_full_naive[
    DT: DType = DType.int32
](
    target: Span[UInt8],
    query: Span[UInt8],
    scoring_matrix: ScoringMatrix,
    *,
    match_score: Scalar[DT] = 1,
    mismatch_score: Scalar[DT] = -1,
    gap_open_penalty: Scalar[DT] = -3,
    gap_extension_penalty: Scalar[DT] = -1,
) -> AlignmentResult:
    """Needleman-Wunsch algorithm for global sequence alignment with affine gap penalties.

    - H: Main scoring matrix (match/mismatc)
    - E: Gap in vertical direction (gap in seq2, (target))
    - F: Gap in horizontal direction (gap in seq1, (query))

    ```mojo
    from testing import assert_equal
    from ishlib.matcher.alignment.global_aln.basic import needleman_wunsch_full_naive
    from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix

    var score_matrix = ScoringMatrix.actgn_matrix()
    var result = needleman_wunsch_full_naive(
        score_matrix.convert_ascii_to_encoding("ATGC".as_bytes()),
        score_matrix.convert_ascii_to_encoding("ATTGCC".as_bytes()),
        score_matrix
    )
    assert_equal(result.score, 2)
    result = needleman_wunsch_full_naive(
        score_matrix.convert_ascii_to_encoding("AAAA".as_bytes()),
        score_matrix.convert_ascii_to_encoding("ATTGCC".as_bytes()),
        score_matrix
    )
    assert_equal(result.score, -8)
    result = needleman_wunsch_full_naive(
        score_matrix.convert_ascii_to_encoding("GATTACA".as_bytes()),
        score_matrix.convert_ascii_to_encoding("GCATGCN".as_bytes()),
        score_matrix
    )
    assert_equal(result.score, 2)
    ```
    """
    alias NUM = Scalar[DT]

    # Init the sizes
    var rows = len(query) + 1
    var cols = len(target) + 1

    # Init the matrices for dynamic programming
    # H: Main scoring matrix (match/mismatch)
    # E: Gap in vertical direction (gap in target)
    # F: Gap in horizontal direction (gap in query)
    var H = List[List[NUM]](capacity=rows)
    var E = List[List[NUM]](capacity=rows)
    var F = List[List[NUM]](capacity=rows)

    for _ in range(0, rows + 1):
        var h_row = List[NUM](capacity=cols)
        var e_row = List[NUM](capacity=cols)
        var f_row = List[NUM](capacity=cols)
        for _ in range(0, cols + 1):
            h_row.append(0)
            e_row.append(0)
            f_row.append(0)
        H.append(h_row)
        E.append(e_row)
        F.append(f_row)

    # Base cases
    H[0][0] = 0  # To left corner is always zero
    E[0][0] = NUM.MIN // 2
    F[0][0] = NUM.MIN // 2

    # Init first row (gaps in the horizontal, (query))
    for j in range(1, cols + 1):
        H[0][j] = NUM.MIN // 2
        E[0][j] = NUM.MIN // 2

        # First gap: gap open
        # Subsequent gaps: previous + extension
        if j == 1:
            F[0][j] = gap_open_penalty
        else:
            F[0][j] = F[0][j - 1] + gap_extension_penalty

    # Init first column (gaps in vertical direction (target))
    for i in range(1, rows + 1):
        H[i][0] = NUM.MIN // 2
        F[i][0] = NUM.MIN // 2

        # First gap: gap open
        # Subsequent gaps: previous + extension
        if i == 1:
            E[i][0] = gap_open_penalty
        else:
            E[i][0] = E[i - 1][0] + gap_extension_penalty

    # Fill in the matrix
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            # Calculate scores for F matrix
            F[i][j] = max(
                H[i][j - 1] + gap_open_penalty,
                F[i][j - 1] + gap_extension_penalty,
            )
            # Calculate scores for E matrix
            E[i][j] = max(
                H[i - 1][j] + gap_open_penalty,
                E[i - 1][j] + gap_extension_penalty,
            )

            # Calculate scores for H matrix
            var current = scoring_matrix.get(
                Int(query[i - 1]), Int(target[j - 1])
            ).cast[DT]()
            # print(
            #     chr(Int(seq1[i - 1])), "vs", chr(Int(seq2[j - 1])), "=", current
            # )
            H[i][j] = max(
                H[i - 1][j - 1] + current,
                max(E[i][j], F[i][j]),
            )

    var score = max(
        H[rows - 1][cols - 1], max(F[rows - 1][cols - 1], E[rows - 1][cols - 1])
    )

    # print("H")
    # for i in range(0, rows + 1):
    #     for j in range(0, cols + 1):
    #         print(H[i][j], ",", end="")
    #     print()

    # print("E")
    # for i in range(0, rows + 1):
    #     for j in range(0, cols + 1):
    #         print(E[i][j], ",", end="")
    #     print()

    # print("F")
    # for i in range(0, rows + 1):
    #     for j in range(0, cols + 1):
    #         print(F[i][j], ",", end="")
    #     print()

    return AlignmentResult(score.cast[DType.int32](), None, None, None)


fn needleman_wunsch_parasail[
    DT: DType = DType.int32
](
    query: Span[UInt8],
    target: Span[UInt8],
    read scoring_matrix: ScoringMatrix,
    *,
    match_score: Scalar[DT] = 1,
    mismatch_score: Scalar[DT] = -1,
    gap_open_penalty: Scalar[DT] = -3,
    gap_extension_penalty: Scalar[DT] = -1,
) -> AlignmentResult:
    """Needleman-Wunsch algorithm for global sequence alignment with affine gap penalties.

    Note: query and target need to have already been encoded by the matrix.

    - H: Main scoring matrix (match/mismatc)
    - E: Gap in vertical direction (gap in seq2, (target))
    - F: Gap in horizontal direction (gap in seq1, (query))

    ```mojo
    from testing import assert_equal
    from ishlib.matcher.alignment.global_aln.basic import needleman_wunsch_parasail
    from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix

    var score_matrix = ScoringMatrix.actgn_matrix()
    var result = needleman_wunsch_parasail(
        score_matrix.convert_ascii_to_encoding("ATGC".as_bytes()),
        score_matrix.convert_ascii_to_encoding("ATTGCC".as_bytes()),
        score_matrix
    )
    assert_equal(result.score, 2)
    result = needleman_wunsch_parasail(
        score_matrix.convert_ascii_to_encoding("AAAA".as_bytes()),
        score_matrix.convert_ascii_to_encoding("ATTGCC".as_bytes()),
        score_matrix
    )
    assert_equal(result.score, -8)
    result = needleman_wunsch_parasail(
        score_matrix.convert_ascii_to_encoding("GATTACA".as_bytes()),
        score_matrix.convert_ascii_to_encoding("GCATGCN".as_bytes()),
        score_matrix
    )
    assert_equal(result.score, 2)
    ```
    """
    alias NUM = Scalar[DT]

    # TODO: can I make it so that the query is in the inner loop? then the memory usage is based on the query?

    # Init the sizes
    var rows = len(target) + 1
    var cols = len(query) + 1

    var H = List[NUM](capacity=cols)
    var F = List[NUM](capacity=cols)
    for _ in range(0, cols):
        H.append(0)
        F.append(0)

    # Upper Left corner
    H[0] = 0
    F[0] = NUM.MIN // 2

    # First Row
    for j in range(1, cols):
        H[j] = gap_open_penalty + ((j - 1) * gap_extension_penalty)
        F[j] = NUM.MIN // 2

    # Iterate over the target sequence
    for i in range(1, rows):
        # Init first column
        var NH = H[0]
        var WH = gap_open_penalty + ((i - 1) * gap_extension_penalty)
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
                Int(query[j - 1]), Int(target[i - 1])
            ).cast[DT]()
            WH = max(max(H_dag, E), F[j])
            H[j] = WH

    var score = H[len(query)]
    return AlignmentResult(
        score.cast[DType.int32](), None, None, coords=(0, len(target))
    )
