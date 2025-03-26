from testing import assert_equal, assert_true
from ishlib.matcher.alignment import AlignmentResult
from ishlib.matcher.alignment.semi_global_aln.basic import (
    semi_global_parasail,
    semi_global_parasail_start_end_end,
)
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix
from ishlib.matcher.alignment.striped_utils import ScoreSize

from ishlib.matcher.alignment.semi_global_aln.striped import (
    semi_global_aln,
    semi_global_aln_start_end,
    Profile,
)


@value
struct FreeGapTest(StringableRaising):
    var q_start: Bool
    var q_end: Bool
    var t_start: Bool
    var t_end: Bool
    var score: Int32

    fn __str__(read self) raises -> String:
        return (
            "q_start: {}, q_end: {}, t_start: {}, t_end: {}, score: {}".format(
                self.q_start, self.q_end, self.t_start, self.t_end, self.score
            )
        )


fn test_exact_match() raises:
    """Test exact match between sequences."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("ACGT".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding("ACGT".as_bytes())

    # Test all combinations of free gaps
    var configs = List[FreeGapTest]()
    # query_start, query_end, target_start, target_end, expected_score
    configs.append(FreeGapTest(False, False, False, False, 8))  # No free gaps
    configs.append(
        FreeGapTest(True, False, False, False, 8)
    )  # Free query start
    configs.append(FreeGapTest(False, True, False, False, 8))  # Free query end
    configs.append(
        FreeGapTest(False, False, True, False, 8)
    )  # Free target start
    configs.append(FreeGapTest(False, False, False, True, 8))  # Free target end
    configs.append(FreeGapTest(True, True, False, False, 8))  # Free query ends
    configs.append(FreeGapTest(False, False, True, True, 8))  # Free target ends
    configs.append(FreeGapTest(True, True, True, True, 8))  # All free

    for i in range(len(configs)):
        var config = configs[i]

        var result = semi_global_parasail(
            query,
            target,
            score_matrix,
            free_query_start_gaps=config.q_start,
            free_query_end_gaps=config.q_end,
            free_target_start_gaps=config.t_start,
            free_target_end_gaps=config.t_end,
        )

        assert_equal(
            result.score,
            config.score,
            "Exact match with gaps (q_start="
            + String(config.q_start)
            + ", q_end="
            + String(config.q_end)
            + ", t_start="
            + String(config.t_start)
            + ", t_end="
            + String(config.t_end)
            + ") should score "
            + String(config.score),
        )

        # For exact matches, end positions should be the last indices
        assert_equal(result.query, 3, "Query end should be at index 3")
        assert_equal(result.target, 3, "Target end should be at index 3")


fn test_exact_match_striped() raises:
    """Test exact match between sequences."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("ACGT".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding("ACGT".as_bytes())

    var profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](query, score_matrix, ScoreSize.Adaptive)

    # Test all combinations of free gaps
    var configs = List[FreeGapTest]()
    # query_start, query_end, target_start, target_end, expected_score
    configs.append(FreeGapTest(False, False, False, False, 8))  # No free gaps
    configs.append(
        FreeGapTest(True, False, False, False, 8)
    )  # Free query start
    configs.append(FreeGapTest(False, True, False, False, 8))  # Free query end
    configs.append(
        FreeGapTest(False, False, True, False, 8)
    )  # Free target start
    configs.append(FreeGapTest(False, False, False, True, 8))  # Free target end
    configs.append(FreeGapTest(True, True, False, False, 8))  # Free query ends
    configs.append(FreeGapTest(False, False, True, True, 8))  # Free target ends
    configs.append(FreeGapTest(True, True, True, True, 8))  # All free

    for i in range(len(configs)):
        var config = configs[i]

        var result = semi_global_aln[DType.uint16, 8](
            target,
            len(query),
            gap_open_penalty=3,
            gap_extension_penalty=1,
            profile=profile.profile_large.value(),
            bias=profile.bias.cast[DType.uint16](),
            free_query_start_gaps=config.q_start,
            free_query_end_gaps=config.q_end,
            free_target_start_gaps=config.t_start,
            free_target_end_gaps=config.t_end,
        ).best

        assert_equal(
            result.score.cast[DType.int32](),
            config.score,
            "Exact match with gaps (q_start="
            + String(config.q_start)
            + ", q_end="
            + String(config.q_end)
            + ", t_start="
            + String(config.t_start)
            + ", t_end="
            + String(config.t_end)
            + ") should score "
            + String(config.score),
        )

        # For exact matches, end positions should be the last indices
        assert_equal(result.query, 3, "Query end should be at index 3")
        assert_equal(result.reference, 3, "Target end should be at index 3")


fn test_query_substring() raises:
    """Test query as substring of target."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("ACGT".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding("TTACGTCC".as_bytes())
    var rev_query = query.copy()
    var rev_target = target.copy()
    rev_query.reverse()
    rev_target.reverse()

    # Test with free target ends (query should align without gaps)
    var result = semi_global_parasail(
        query,
        target,
        score_matrix,
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=True,
        free_target_end_gaps=True,
    )

    assert_equal(result.score, 8, "Query as substring should score 8")
    assert_equal(result.query, 3, "Query end should be at index 3")
    assert_equal(result.target, 5, "Target end should be at index 5")

    # Test with start/end detection
    var alignment = semi_global_parasail_start_end_end(
        query,
        target,
        rev_query,
        rev_target,
        score_matrix,
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=True,
        free_target_end_gaps=True,
    )

    assert_equal(
        alignment.score,
        8,
        "Query as substring with start/end detection should score 8",
    )
    assert_equal(
        alignment.coords.value().start,
        2,
        "Alignment should start at target index 2",
    )
    assert_equal(
        alignment.coords.value().end,
        5,
        "Alignment should end at target index 5",
    )


fn test_query_substring_striped() raises:
    """Test query as substring of target."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("ACGT".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding("TTACGTCC".as_bytes())
    var rev_query = query.copy()
    var rev_target = target.copy()
    rev_query.reverse()
    rev_target.reverse()
    var profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](query, score_matrix, ScoreSize.Adaptive)
    var rev_profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](rev_query, score_matrix, ScoreSize.Adaptive)

    # Test with free target ends (query should align without gaps)
    var result = semi_global_aln[DType.uint16, 8](
        target,
        len(query),
        gap_open_penalty=3,
        gap_extension_penalty=1,
        profile=profile.profile_large.value(),
        bias=profile.bias.cast[DType.uint16](),
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=True,
        free_target_end_gaps=True,
    ).best

    assert_equal(result.score, 8, "Query as substring should score 8")
    assert_equal(result.query, 3, "Query end should be at index 3")
    assert_equal(result.reference, 5, "Target end should be at index 5")

    # Test with start/end detection
    var alignment = semi_global_aln_start_end[DType.uint16, 8](
        reference=target,
        rev_reference=rev_target,
        query_len=len(query),
        gap_open_penalty=3,
        gap_extension_penalty=1,
        profile=profile.profile_large.value(),
        rev_profile=rev_profile.profile_large.value(),
        bias=profile.bias.cast[DType.uint16](),
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=True,
        free_target_end_gaps=True,
    )

    assert_equal(
        alignment.score,
        8,
        "Query as substring with start/end detection should score 8",
    )
    assert_equal(
        alignment.target_start,
        2,
        "Alignment should start at target index 2",
    )
    assert_equal(
        alignment.target_end,
        5,
        "Alignment should end at target index 5",
    )


fn test_target_substring() raises:
    """Test target as substring of query."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("GGACGTAA".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding("ACGT".as_bytes())
    var rev_query = query.copy()
    var rev_target = target.copy()
    rev_query.reverse()
    rev_target.reverse()
    var profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](query, score_matrix, ScoreSize.Adaptive)
    var rev_profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](rev_query, score_matrix, ScoreSize.Adaptive)

    # Test with free query ends (target should align without gaps)
    var result = semi_global_aln[DType.uint16, 8](
        target,
        len(query),
        gap_open_penalty=3,
        gap_extension_penalty=1,
        profile=profile.profile_large.value(),
        bias=profile.bias.cast[DType.uint16](),
        free_query_start_gaps=True,
        free_query_end_gaps=True,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    ).best

    assert_equal(result.score, 8, "Target as substring should score 8")
    assert_equal(result.query, 5, "Query end should be at index 5")
    assert_equal(result.reference, 3, "Target end should be at index 3")

    # Test with start/end detection
    var alignment = semi_global_aln_start_end[DType.uint16, 8](
        reference=target,
        rev_reference=rev_target,
        query_len=len(query),
        gap_open_penalty=3,
        gap_extension_penalty=1,
        profile=profile.profile_large.value(),
        rev_profile=rev_profile.profile_large.value(),
        bias=profile.bias.cast[DType.uint16](),
        free_query_start_gaps=True,
        free_query_end_gaps=True,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    )

    assert_equal(
        alignment.score,
        8,
        "Target as substring with start/end detection should score 8",
    )
    assert_equal(
        alignment.target_start,
        0,
        "Alignment should start at target index 0",
    )
    assert_equal(
        alignment.target_end,
        3,
        "Alignment should end at target index 3",
    )


fn test_target_substring_striped() raises:
    """Test target as substring of query."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("GGACGTAA".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding("ACGT".as_bytes())
    var rev_query = query.copy()
    var rev_target = target.copy()
    rev_query.reverse()
    rev_target.reverse()

    # Test with free query ends (target should align without gaps)
    var result = semi_global_parasail(
        query,
        target,
        score_matrix,
        free_query_start_gaps=True,
        free_query_end_gaps=True,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    )

    assert_equal(result.score, 8, "Target as substring should score 8")
    assert_equal(result.query, 5, "Query end should be at index 5")
    assert_equal(result.target, 3, "Target end should be at index 3")

    # Test with start/end detection
    var alignment = semi_global_parasail_start_end_end(
        query,
        target,
        rev_query,
        rev_target,
        score_matrix,
        free_query_start_gaps=True,
        free_query_end_gaps=True,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    )

    assert_equal(
        alignment.score,
        8,
        "Target as substring with start/end detection should score 8",
    )
    assert_equal(
        alignment.coords.value().start,
        0,
        "Alignment should start at target index 0",
    )
    assert_equal(
        alignment.coords.value().end,
        3,
        "Alignment should end at target index 3",
    )


fn test_partial_match_with_mismatch() raises:
    """Test partial match with mismatch."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("ACGT".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding(
        "ACTT".as_bytes()
    )  # Last base is different

    # No free ends config
    var result = semi_global_parasail(
        query,
        target,
        score_matrix,
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    )

    # From parasail
    # optimal_alignment_score: 4      strand: +       target_begin: 1 target_end: 4   query_begin: 1  query_end: 4
    # Target:          1 ACGT       4
    #                    ||*|
    # Query:           1 ACTT       4

    assert_equal(result.score, 4, "Partial match with mismatch should score 4")

    # With free query end - should still align through the mismatch
    result = semi_global_parasail(
        query,
        target,
        score_matrix,
        free_query_start_gaps=False,
        free_query_end_gaps=True,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    )

    assert_equal(
        result.score,
        4,
        "Partial match with mismatch and free query end should score 5",
    )


fn test_alignment_with_gap() raises:
    """Test alignment with a gap."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("ACGTT".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding(
        "ACTT".as_bytes()
    )  # G is missing
    var rev_query = query.copy()
    var rev_target = target.copy()
    rev_query.reverse()
    rev_target.reverse()

    # No free ends config
    var result = semi_global_parasail(
        query,
        target,
        score_matrix,
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    )

    assert_equal(result.score, 5, "Alignment with gap should score 5")

    # Test with start/end detection
    var alignment = semi_global_parasail_start_end_end(
        query,
        target,
        rev_query,
        rev_target,
        score_matrix,
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    )

    assert_equal(alignment.score, 5, "Alignment with gap should score 5")


fn test_no_match() raises:
    """Test case with no matching characters."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("AAAA".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding("CCCC".as_bytes())

    # With all ends free
    var result = semi_global_parasail(
        query,
        target,
        score_matrix,
        free_query_start_gaps=True,
        free_query_end_gaps=True,
        free_target_start_gaps=True,
        free_target_end_gaps=True,
    )

    # From parasail
    # optimal_alignment_score: -2     strand: +       target_begin: 1 target_end: 1   query_begin: 1  query_end: 4
    # Target:          1 ---AAAA       4
    #                       *
    # Query:           1 CCCC---       4

    assert_equal(
        result.score, -2, "No match with all free ends should score -2"
    )

    # With no free ends
    result = semi_global_parasail(
        query,
        target,
        score_matrix,
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    )

    assert_equal(result.score, -8, "No match with no free ends should score -8")


fn test_empty_sequences() raises:
    """Test with empty sequences."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var empty = score_matrix.convert_ascii_to_encoding("".as_bytes())
    var empty_clone = empty.copy()
    var nonempty = score_matrix.convert_ascii_to_encoding("ACGT".as_bytes())

    # One empty sequence with free start/end gaps on query
    var result = semi_global_parasail(
        empty,
        nonempty,
        score_matrix,
        free_query_start_gaps=True,
        free_query_end_gaps=True,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    )

    assert_equal(
        result.score,
        0,
        "Empty query with free ends should score 0",
    )

    # Both empty sequences
    var ret = semi_global_parasail(
        empty,
        empty_clone,
        score_matrix,
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    )

    assert_equal(ret.score, 0, "Both empty sequences should score -2")


fn test_complex_case() raises:
    """Test a more complex alignment scenario."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("AGTACGACGT".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding("TATCGTACGT".as_bytes())
    var rev_query = query.copy()
    var rev_target = target.copy()
    rev_query.reverse()
    rev_target.reverse()

    # Test with various free end configurations
    var configs = List[FreeGapTest]()
    # query_start, query_end, target_start, target_end, expected_score
    configs.append(FreeGapTest(False, False, False, False, 6))  # No free gaps
    configs.append(
        FreeGapTest(True, False, False, False, 10)
    )  # Free query start
    configs.append(FreeGapTest(False, True, False, False, 6))  # Free query end
    configs.append(
        FreeGapTest(False, False, True, False, 7)
    )  # Free target start
    configs.append(FreeGapTest(False, False, False, True, 6))  # Free target end
    configs.append(FreeGapTest(True, True, False, False, 10))  # Free query ends
    configs.append(FreeGapTest(False, False, True, True, 7))  # Free target ends
    configs.append(FreeGapTest(True, True, True, True, 10))  # All free

    for i in range(len(configs)):
        var config = configs[i]

        var result = semi_global_parasail(
            query,
            target,
            score_matrix,
            free_query_start_gaps=config.q_start,
            free_query_end_gaps=config.q_end,
            free_target_start_gaps=config.t_start,
            free_target_end_gaps=config.t_end,
        )

        assert_equal(
            result.score,
            config.score,
            "Complex case with gaps (q_start="
            + String(config.q_start)
            + ", q_end="
            + String(config.q_end)
            + ", t_start="
            + String(config.t_start)
            + ", t_end="
            + String(config.t_end)
            + ") should score "
            + String(config.score),
        )

        # Test start/end detection for free query ends
        if (
            config.q_start
            and config.q_end
            and not config.t_start
            and not config.t_end
        ):
            var alignment = semi_global_parasail_start_end_end(
                query,
                target,
                rev_query,
                rev_target,
                score_matrix,
                free_query_start_gaps=config.q_start,
                free_query_end_gaps=config.q_end,
                free_target_start_gaps=config.t_start,
                free_target_end_gaps=config.t_end,
            )

            assert_equal(
                alignment.score,
                config.score,
                "Complex case start/end should match score",
            )
            assert_true(
                alignment.coords.value().start >= 0,
                "Start position should be valid",
            )
            assert_true(
                alignment.coords.value().end <= len(target),
                "End position should be valid",
            )
            assert_true(
                alignment.coords.value().start <= alignment.coords.value().end,
                "Start should be <= end",
            )


fn test_reversed_alignment() raises:
    """Test that the reversed alignment correctly identifies start position."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("TTACGTCC".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding("ACGT".as_bytes())
    var rev_query = query.copy()
    var rev_target = target.copy()
    rev_query.reverse()
    rev_target.reverse()

    # The alignment should find target within query
    var alignment = semi_global_parasail_start_end_end(
        query,
        target,
        rev_query,
        rev_target,
        score_matrix,
        free_query_start_gaps=True,
        free_query_end_gaps=True,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    )
    # reminder: all coords are relative to the target
    assert_equal(alignment.score, 8, "Reversed alignment should score 8")
    assert_equal(
        alignment.coords.value().start,
        0,
        "Alignment should start at query index 0",
    )
    assert_equal(
        alignment.coords.value().end, 3, "Alignment should end at query index 5"
    )


fn test_biological_example() raises:
    """Test with biologically relevant examples."""
    var score_matrix = ScoringMatrix.actgn_matrix()

    # Test gene finding at beginning of genome
    var gene = score_matrix.convert_ascii_to_encoding(
        "ATGGCGTGCAATG".as_bytes()
    )
    var genome = score_matrix.convert_ascii_to_encoding(
        "ATGGCGTGCAATGCCCGGTACGT".as_bytes()
    )
    var rev_gene = gene.copy()
    var rev_genome = genome.copy()
    rev_gene.reverse()
    rev_genome.reverse()

    var result = semi_global_parasail(
        gene,
        genome,
        score_matrix,
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=False,
        free_target_end_gaps=True,
    )

    assert_equal(result.score, 26, "Gene at beginning should score 26")

    # Find gene in middle of genome
    gene = score_matrix.convert_ascii_to_encoding("CCCGGT".as_bytes())
    rev_gene = gene.copy()
    rev_gene.reverse()

    # ATGGCGTGCAATGCCCGGTACGT
    # -------------CCCGGT----
    var alignment = semi_global_parasail_start_end_end(
        gene,
        genome,
        rev_gene,
        rev_genome,
        score_matrix,
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=True,
        free_target_end_gaps=True,
    )
    assert_equal(alignment.score, 12, "Middle gene should score 12")
    assert_equal(
        alignment.coords.value().start,
        13,
        "Middle gene should start at index 13",
    )
    assert_equal(
        alignment.coords.value().end, 18, "Middle gene should end at index 18"
    )

    # Find gene at end of genome
    gene = score_matrix.convert_ascii_to_encoding("TACGT".as_bytes())
    rev_gene = gene.copy()
    rev_gene.reverse()

    alignment = semi_global_parasail_start_end_end(
        gene,
        genome,
        rev_gene,
        rev_genome,
        score_matrix,
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=True,
        free_target_end_gaps=False,
    )
    # ATGGCGTGCAATGCCCGGTACGT
    # ------------------TACGT
    assert_equal(alignment.score, 10, "End gene should score 10")
    assert_equal(
        alignment.coords.value().start, 18, "End gene should start at index 18"
    )
    assert_equal(
        alignment.coords.value().end, 22, "End gene should end at index 22"
    )


fn test_partial_match_with_mismatch_striped() raises:
    """Test partial match with mismatch."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("ACGT".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding(
        "ACTT".as_bytes()
    )  # Last base is different
    var profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](query, score_matrix, ScoreSize.Adaptive)

    # No free ends config
    var result = semi_global_aln[DType.uint16, 8](
        target,
        len(query),
        gap_open_penalty=3,
        gap_extension_penalty=1,
        profile=profile.profile_large.value(),
        bias=profile.bias.cast[DType.uint16](),
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    ).best

    assert_equal(result.score, 4, "Partial match with mismatch should score 4")

    # With free query end - should still align through the mismatch
    result = semi_global_aln[DType.uint16, 8](
        target,
        len(query),
        gap_open_penalty=3,
        gap_extension_penalty=1,
        profile=profile.profile_large.value(),
        bias=profile.bias.cast[DType.uint16](),
        free_query_start_gaps=False,
        free_query_end_gaps=True,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    ).best

    assert_equal(
        result.score,
        4,
        "Partial match with mismatch and free query end should score 4",
    )


fn test_alignment_with_gap_striped() raises:
    """Test alignment with a gap."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("ACGTT".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding(
        "ACTT".as_bytes()
    )  # G is missing
    var rev_query = query.copy()
    var rev_target = target.copy()
    rev_query.reverse()
    rev_target.reverse()
    var profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](query, score_matrix, ScoreSize.Adaptive)
    var rev_profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](rev_query, score_matrix, ScoreSize.Adaptive)

    # No free ends config
    var result = semi_global_aln[DType.uint16, 8](
        target,
        len(query),
        gap_open_penalty=3,
        gap_extension_penalty=1,
        profile=profile.profile_large.value(),
        bias=profile.bias.cast[DType.uint16](),
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    ).best

    assert_equal(result.score, 5, "Alignment with gap should score 5")

    # Test with start/end detection
    var alignment = semi_global_aln_start_end[DType.uint16, 8](
        reference=target,
        rev_reference=rev_target,
        query_len=len(query),
        gap_open_penalty=3,
        gap_extension_penalty=1,
        profile=profile.profile_large.value(),
        rev_profile=rev_profile.profile_large.value(),
        bias=profile.bias.cast[DType.uint16](),
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    )

    assert_equal(alignment.score, 5, "Alignment with gap should score 5")


fn test_no_match_striped() raises:
    """Test case with no matching characters."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("AAAA".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding("CCCC".as_bytes())
    var profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](query, score_matrix, ScoreSize.Adaptive)

    # With all ends free
    var result = semi_global_aln[DType.uint16, 8](
        target,
        len(query),
        gap_open_penalty=3,
        gap_extension_penalty=1,
        profile=profile.profile_large.value(),
        bias=profile.bias.cast[DType.uint16](),
        free_query_start_gaps=True,
        free_query_end_gaps=True,
        free_target_start_gaps=True,
        free_target_end_gaps=True,
    ).best

    assert_equal(
        result.score, -2, "No match with all free ends should score -2"
    )

    # With no free ends
    result = semi_global_aln[DType.uint16, 8](
        target,
        len(query),
        gap_open_penalty=3,
        gap_extension_penalty=1,
        profile=profile.profile_large.value(),
        bias=profile.bias.cast[DType.uint16](),
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    ).best

    assert_equal(result.score, -8, "No match with no free ends should score -8")


fn test_empty_sequences_striped() raises:
    """Test with empty sequences."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var empty = score_matrix.convert_ascii_to_encoding("".as_bytes())
    var empty_clone = empty.copy()
    var nonempty = score_matrix.convert_ascii_to_encoding("ACGT".as_bytes())
    var profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](empty, score_matrix, ScoreSize.Adaptive)

    # One empty sequence with free start/end gaps on query
    var result = semi_global_aln[DType.uint16, 8](
        nonempty,
        len(empty),
        gap_open_penalty=3,
        gap_extension_penalty=1,
        profile=profile.profile_large.value(),
        bias=profile.bias.cast[DType.uint16](),
        free_query_start_gaps=True,
        free_query_end_gaps=True,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    ).best

    assert_equal(
        result.score,
        0,
        "Empty query with free ends should score 0",
    )

    # Both empty sequences
    var empty_profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](empty, score_matrix, ScoreSize.Adaptive)

    var ret = semi_global_aln[DType.uint16, 8](
        empty_clone,
        len(empty),
        gap_open_penalty=3,
        gap_extension_penalty=1,
        profile=empty_profile.profile_large.value(),
        bias=empty_profile.bias.cast[DType.uint16](),
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    ).best

    assert_equal(ret.score, 0, "Both empty sequences should score 0")


fn test_complex_case_striped() raises:
    """Test a more complex alignment scenario."""

    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("AGTACGACGT".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding("TATCGTACGT".as_bytes())
    var rev_query = query.copy()
    var rev_target = target.copy()
    rev_query.reverse()
    rev_target.reverse()
    var profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](query, score_matrix, ScoreSize.Adaptive)
    var rev_profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](rev_query, score_matrix, ScoreSize.Adaptive)

    # Test with various free end configurations
    var configs = List[FreeGapTest]()
    # query_start, query_end, target_start, target_end, expected_score
    configs.append(FreeGapTest(False, False, False, False, 6))  # No free gaps
    configs.append(
        FreeGapTest(True, False, False, False, 10)
    )  # Free query start
    configs.append(FreeGapTest(False, True, False, False, 6))  # Free query end
    configs.append(
        FreeGapTest(False, False, True, False, 7)
    )  # Free target start
    configs.append(FreeGapTest(False, False, False, True, 6))  # Free target end
    configs.append(FreeGapTest(True, True, False, False, 10))  # Free query ends
    configs.append(FreeGapTest(False, False, True, True, 7))  # Free target ends
    configs.append(FreeGapTest(True, True, True, True, 10))  # All free

    for i in range(len(configs)):
        var config = configs[i]

        var result = semi_global_aln[DType.uint16, 8](
            target,
            len(query),
            gap_open_penalty=3,
            gap_extension_penalty=1,
            profile=profile.profile_large.value(),
            bias=profile.bias.cast[DType.uint16](),
            free_query_start_gaps=config.q_start,
            free_query_end_gaps=config.q_end,
            free_target_start_gaps=config.t_start,
            free_target_end_gaps=config.t_end,
        ).best

        assert_equal(
            result.score,
            config.score,
            "Complex case with gaps (q_start="
            + String(config.q_start)
            + ", q_end="
            + String(config.q_end)
            + ", t_start="
            + String(config.t_start)
            + ", t_end="
            + String(config.t_end)
            + ") should score "
            + String(config.score),
        )

        # Test start/end detection for free query ends
        if (
            config.q_start
            and config.q_end
            and not config.t_start
            and not config.t_end
        ):
            var alignment = semi_global_aln_start_end[DType.uint16, 8](
                reference=target,
                rev_reference=rev_target,
                query_len=len(query),
                gap_open_penalty=3,
                gap_extension_penalty=1,
                profile=profile.profile_large.value(),
                rev_profile=rev_profile.profile_large.value(),
                bias=profile.bias.cast[DType.uint16](),
                free_query_start_gaps=config.q_start,
                free_query_end_gaps=config.q_end,
                free_target_start_gaps=config.t_start,
                free_target_end_gaps=config.t_end,
            )

            assert_equal(
                alignment.score,
                config.score,
                "Complex case start/end should match score",
            )
            assert_true(
                alignment.target_start >= 0,
                "Start position should be valid",
            )
            assert_true(
                alignment.target_end <= len(target),
                "End position should be valid",
            )
            assert_true(
                alignment.target_start <= alignment.target_end,
                "Start should be <= end",
            )


fn test_complex_case_striped_small() raises:
    """Test a more complex alignment scenario."""

    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("AGTACGACGT".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding("TATCGTACGT".as_bytes())
    var rev_query = query.copy()
    var rev_target = target.copy()
    rev_query.reverse()
    rev_target.reverse()
    var profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](query, score_matrix, ScoreSize.Adaptive)
    var rev_profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](rev_query, score_matrix, ScoreSize.Adaptive)

    # Test with various free end configurations
    var configs = List[FreeGapTest]()
    # query_start, query_end, target_start, target_end, expected_score
    configs.append(FreeGapTest(False, False, False, False, 6))  # No free gaps
    configs.append(
        FreeGapTest(True, False, False, False, 10)
    )  # Free query start
    configs.append(FreeGapTest(False, True, False, False, 6))  # Free query end
    configs.append(
        FreeGapTest(False, False, True, False, 7)
    )  # Free target start
    configs.append(FreeGapTest(False, False, False, True, 6))  # Free target end
    configs.append(FreeGapTest(True, True, False, False, 10))  # Free query ends
    configs.append(FreeGapTest(False, False, True, True, 7))  # Free target ends
    configs.append(FreeGapTest(True, True, True, True, 10))  # All free

    for i in range(len(configs)):
        var config = configs[i]

        var result = semi_global_aln[DType.uint8, 16](
            target,
            len(query),
            gap_open_penalty=3,
            gap_extension_penalty=1,
            profile=profile.profile_small.value(),
            bias=profile.bias.cast[DType.uint8](),
            free_query_start_gaps=config.q_start,
            free_query_end_gaps=config.q_end,
            free_target_start_gaps=config.t_start,
            free_target_end_gaps=config.t_end,
        ).best

        assert_equal(
            result.score,
            config.score,
            "Complex case with gaps (q_start="
            + String(config.q_start)
            + ", q_end="
            + String(config.q_end)
            + ", t_start="
            + String(config.t_start)
            + ", t_end="
            + String(config.t_end)
            + ") should score "
            + String(config.score),
        )

        # Test start/end detection for free query ends
        if (
            config.q_start
            and config.q_end
            and not config.t_start
            and not config.t_end
        ):
            var alignment = semi_global_aln_start_end[DType.uint8, 16](
                reference=target,
                rev_reference=rev_target,
                query_len=len(query),
                gap_open_penalty=3,
                gap_extension_penalty=1,
                profile=profile.profile_small.value(),
                rev_profile=rev_profile.profile_small.value(),
                bias=profile.bias.cast[DType.uint8](),
                free_query_start_gaps=config.q_start,
                free_query_end_gaps=config.q_end,
                free_target_start_gaps=config.t_start,
                free_target_end_gaps=config.t_end,
            )

            assert_equal(
                alignment.score,
                config.score,
                "Complex case start/end should match score",
            )
            assert_true(
                alignment.target_start >= 0,
                "Start position should be valid",
            )
            assert_true(
                alignment.target_end <= len(target),
                "End position should be valid",
            )
            assert_true(
                alignment.target_start <= alignment.target_end,
                "Start should be <= end",
            )


fn test_reversed_alignment_striped() raises:
    """Test that the reversed alignment correctly identifies start position."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var query = score_matrix.convert_ascii_to_encoding("TTACGTCC".as_bytes())
    var target = score_matrix.convert_ascii_to_encoding("ACGT".as_bytes())
    var rev_query = query.copy()
    var rev_target = target.copy()
    rev_query.reverse()
    rev_target.reverse()
    var profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](query, score_matrix, ScoreSize.Adaptive)
    var rev_profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](rev_query, score_matrix, ScoreSize.Adaptive)

    # The alignment should find target within query
    var alignment = semi_global_aln_start_end[DType.uint16, 8](
        reference=target,
        rev_reference=rev_target,
        query_len=len(query),
        gap_open_penalty=3,
        gap_extension_penalty=1,
        profile=profile.profile_large.value(),
        rev_profile=rev_profile.profile_large.value(),
        bias=profile.bias.cast[DType.uint16](),
        free_query_start_gaps=True,
        free_query_end_gaps=True,
        free_target_start_gaps=False,
        free_target_end_gaps=False,
    )

    assert_equal(alignment.score, 8, "Reversed alignment should score 8")
    assert_equal(
        alignment.target_start,
        0,
        "Alignment should start at target index 0",
    )
    assert_equal(
        alignment.target_end, 3, "Alignment should end at target index 3"
    )


fn test_biological_example_striped() raises:
    """Test with biologically relevant examples."""
    var score_matrix = ScoringMatrix.actgn_matrix()

    # Test gene finding at beginning of genome
    var gene = score_matrix.convert_ascii_to_encoding(
        "ATGGCGTGCAATG".as_bytes()
    )
    var genome = score_matrix.convert_ascii_to_encoding(
        "ATGGCGTGCAATGCCCGGTACGT".as_bytes()
    )
    var rev_gene = gene.copy()
    var rev_genome = genome.copy()
    rev_gene.reverse()
    rev_genome.reverse()
    var profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](gene, score_matrix, ScoreSize.Adaptive)

    var result = semi_global_aln[DType.uint16, 8](
        genome,
        len(gene),
        gap_open_penalty=3,
        gap_extension_penalty=1,
        profile=profile.profile_large.value(),
        bias=profile.bias.cast[DType.uint16](),
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=False,
        free_target_end_gaps=True,
    ).best

    assert_equal(result.score, 26, "Gene at beginning should score 26")

    # Find gene in middle of genome
    gene = score_matrix.convert_ascii_to_encoding("CCCGGT".as_bytes())
    rev_gene = gene.copy()
    rev_gene.reverse()
    var mid_profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](gene, score_matrix, ScoreSize.Adaptive)
    var mid_rev_profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](rev_gene, score_matrix, ScoreSize.Adaptive)

    # ATGGCGTGCAATGCCCGGTACGT
    # -------------CCCGGT----
    var alignment = semi_global_aln_start_end[DType.uint16, 8](
        reference=genome,
        rev_reference=rev_genome,
        query_len=len(gene),
        gap_open_penalty=3,
        gap_extension_penalty=1,
        profile=mid_profile.profile_large.value(),
        rev_profile=mid_rev_profile.profile_large.value(),
        bias=mid_profile.bias.cast[DType.uint16](),
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=True,
        free_target_end_gaps=True,
    )
    assert_equal(alignment.score, 12, "Middle gene should score 12")
    assert_equal(
        alignment.target_start,
        13,
        "Middle gene should start at index 13",
    )
    assert_equal(alignment.target_end, 18, "Middle gene should end at index 18")

    # Find gene at end of genome
    gene = score_matrix.convert_ascii_to_encoding("TACGT".as_bytes())
    rev_gene = gene.copy()
    rev_gene.reverse()
    var end_profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](gene, score_matrix, ScoreSize.Adaptive)
    var end_rev_profile = Profile[
        16, 8, SmallType = DType.uint8, LargeType = DType.uint16
    ](rev_gene, score_matrix, ScoreSize.Adaptive)

    alignment = semi_global_aln_start_end[DType.uint16, 8](
        reference=genome,
        rev_reference=rev_genome,
        query_len=len(gene),
        gap_open_penalty=3,
        gap_extension_penalty=1,
        profile=end_profile.profile_large.value(),
        rev_profile=end_rev_profile.profile_large.value(),
        bias=end_profile.bias.cast[DType.uint16](),
        free_query_start_gaps=False,
        free_query_end_gaps=False,
        free_target_start_gaps=True,
        free_target_end_gaps=False,
    )
    # ATGGCGTGCAATGCCCGGTACGT
    # ------------------TACGT
    assert_equal(alignment.score, 10, "End gene should score 10")
    assert_equal(
        alignment.target_start, 18, "End gene should start at index 18"
    )
    assert_equal(alignment.target_end, 22, "End gene should end at index 22")


fn run_striped_tests() raises:
    """Run all the striped implementation tests."""
    test_exact_match_striped()
    test_query_substring_striped()
    test_target_substring_striped()
    test_partial_match_with_mismatch_striped()
    test_alignment_with_gap_striped()
    test_no_match_striped()
    test_empty_sequences_striped()
    test_complex_case_striped()
    test_complex_case_striped_small()
    test_reversed_alignment_striped()
    test_biological_example_striped()
    print("Success on all striped tests!")


fn run_all_tests() raises:
    """Run all semi-global alignment tests and report results."""
    print("Running all Semi-Global alignment tests...")

    test_exact_match()
    test_exact_match_striped()
    test_query_substring()
    test_target_substring()
    test_partial_match_with_mismatch()
    test_alignment_with_gap()
    test_no_match()
    test_empty_sequences()
    test_complex_case()
    test_reversed_alignment()
    test_biological_example()
    run_striped_tests()
    print("All semi-global alignment tests passed successfully!")


# Run the tests if executed directly
fn main() raises:
    run_all_tests()
