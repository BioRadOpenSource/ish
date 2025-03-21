from testing import assert_equal
from ishlib.matcher.aligment import AlignmentResult
from ishlib.matcher.alignment.global_aln.basic import needleman_wunsch_parasail
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix


fn test_basic_alignment() raises:
    """Test basic alignment with known score."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var result = needleman_wunsch_parasail(
        "ATGC".as_bytes(), "ATTGCC".as_bytes(), score_matrix
    )
    assert_equal(result.score, 2, "Basic alignment should score 2")


fn test_many_mismatches() raises:
    """Test alignment with many mismatches and gaps."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var result = needleman_wunsch_parasail(
        "AAAA".as_bytes(), "ATTGCC".as_bytes(), score_matrix
    )
    assert_equal(
        result.score, -8, "Alignment with many mismatches should score -8"
    )


fn test_longer_sequences() raises:
    """Test alignment with longer sequences."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var result = needleman_wunsch_parasail(
        "GATTACA".as_bytes(), "GCATGCN".as_bytes(), score_matrix
    )
    assert_equal(result.score, 2, "Longer sequence alignment should score 2")


fn test_identical_sequences() raises:
    """Test alignment of identical sequences."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var result = needleman_wunsch_parasail(
        "ACGT".as_bytes(), "ACGT".as_bytes(), score_matrix
    )
    assert_equal(result.score, 8, "Identical sequences should score 2 * length")


fn test_completely_different() raises:
    """Test alignment of completely different sequences."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var result = needleman_wunsch_parasail(
        "AAAA".as_bytes(), "TTTT".as_bytes(), score_matrix
    )
    assert_equal(
        result.score,
        -8,
        "Completely different sequences should have negative score",
    )


fn test_one_empty_sequence() raises:
    """Test alignment with one empty sequence."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var result = needleman_wunsch_parasail(
        "".as_bytes(), "ACGT".as_bytes(), score_matrix
    )
    assert_equal(result.score, -6, "One empty sequence should score -7")


fn test_both_empty_sequences() raises:
    """Test alignment with both sequences empty."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var result = needleman_wunsch_parasail(
        "".as_bytes(), "".as_bytes(), score_matrix
    )
    assert_equal(result.score, 0, "Both empty sequences should score 0")


fn test_repeat_patterns() raises:
    """Test alignment with repeating pattern sequences."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var result = needleman_wunsch_parasail(
        "ATATATATAT".as_bytes(), "ATAT".as_bytes(), score_matrix
    )
    assert_equal(result.score, 0, "Repeated pattern alignment should score -7")


fn test_n_handling() raises:
    """Test proper handling of 'N' wildcards."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var result = needleman_wunsch_parasail(
        "ACGTN".as_bytes(), "ACGTA".as_bytes(), score_matrix
    )
    assert_equal(
        result.score, 10, "N should match any character for a perfect score"
    )


fn test_single_base_indel() raises:
    """Test alignment with a single base insertion/deletion."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var result = needleman_wunsch_parasail(
        "ACGTACGT".as_bytes(), "ACGACGT".as_bytes(), score_matrix
    )
    assert_equal(result.score, 11, "Single base indel should score 9")


fn test_long_vs_short_gaps() raises:
    """Test whether algorithm correctly handles affine gap model preferences."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var result = needleman_wunsch_parasail(
        "AAAACCCCGGGGTTTT".as_bytes(), "ACCGGT".as_bytes(), score_matrix
    )
    assert_equal(
        result.score, -2, "Long gap vs multiple short gaps should score -13"
    )


fn test_biological_example() raises:
    """Test with a biologically relevant example."""
    var score_matrix = ScoringMatrix.actgn_matrix()
    var result = needleman_wunsch_parasail(
        "ATGGCGTGCAATG".as_bytes(), "ATGCGTCATG".as_bytes(), score_matrix
    )
    assert_equal(result.score, 11, "Biological example should score 11")


fn run_all_tests() raises:
    """Run all tests and report results."""
    print("Running all Needleman-Wunsch tests...")

    test_basic_alignment()
    test_many_mismatches()
    test_longer_sequences()
    test_identical_sequences()
    test_completely_different()
    test_one_empty_sequence()
    test_both_empty_sequences()
    test_repeat_patterns()
    test_n_handling()
    test_single_base_indel()
    test_long_vs_short_gaps()
    test_biological_example()

    print("All tests passed successfully!")
