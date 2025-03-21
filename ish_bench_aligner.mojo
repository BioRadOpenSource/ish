"""Test program for SSW."""
from sys import stdout, stderr
from sys.info import simdwidthof, has_avx512f
from sys.param_env import env_get_int
from time.time import perf_counter

from ExtraMojo.cli.parser import OptParser, OptConfig, OptKind, ParsedOpts
from ExtraMojo.io.buffered import BufferedWriter
from ExtraMojo.io.delimited import DelimWriter, ToDelimited
from ishlib.matcher.alignment.local_aln.striped import (
    ssw_align,
    Profile,
    ScoreSize,
    Alignment,
)
from ishlib.matcher.alignment.scoring_matrix import ScoringMatrix
from ishlib.formats.fasta import FastaRecord

# Force half width vectors in the case of avx512 since avx 2 seems faster up to around 700len queries.
# alias SIMD_U8_WIDTH = simdwidthof[
#     UInt8
# ]() if not has_avx512f() else simdwidthof[UInt8]() // 2
# alias SIMD_U16_WIDTH = simdwidthof[
#     UInt16
# ]() if not has_avx512f() else simdwidthof[UInt16]() // 2

alias SIMD_MOD = env_get_int["SIMD_MOD", 1]()
"""Modify the SIMD width based on a CLI argument."""

alias FULL_SIMD_U8_WIDTH = simdwidthof[UInt8]()
alias FULL_SIMD_U16_WIDTH = simdwidthof[UInt16]()

alias SIMD_U8_WIDTH = simdwidthof[UInt8]() // SIMD_MOD
alias SIMD_U16_WIDTH = simdwidthof[UInt16]() // SIMD_MOD


fn parse_args() raises -> ParsedOpts:
    var parser = OptParser(
        name="ssw",
        description=(
            "A aligner for benchmarking differing alignment"
            " algorithms.\nSpecifically will read in both the query and target"
            " fastqs in to memory, do any needed prep, then time the alignment"
            " of query vs target."
        ),
    )
    parser.add_opt(
        OptConfig(
            "match-score",
            OptKind.IntLike,
            default_value=String("2"),
            description=(
                "Score for a match. Only used when no matrix is provided."
            ),
        )
    )
    parser.add_opt(
        OptConfig(
            "mismatch-score",
            OptKind.IntLike,
            default_value=String("2"),
            description=(
                "Score for a mismatch, as a positive number that will be"
                " subtracted. Only used when no matrix is provided."
            ),
        )
    )
    parser.add_opt(
        OptConfig(
            "gap-open-score",
            OptKind.IntLike,
            default_value=String("3"),
            description=(
                "Score for a opening a gap, as a positive number that will be"
                " subtracted."
            ),
        )
    )
    parser.add_opt(
        OptConfig(
            "gap-ext-score",
            OptKind.IntLike,
            default_value=String("1"),
            description=(
                "Score for a extending a gap, as a positive number that will be"
                " subtracted."
            ),
        )
    )
    parser.add_opt(
        OptConfig(
            "scoring-matrix",
            OptKind.StringLike,
            default_value=String("Blosum50"),
            description=(
                "Scoring matrix to use. Currently supports: [Blosum50,"
                " Blosum62, ACTGN]"
            ),
        )
    )
    parser.add_opt(
        OptConfig(
            "target-fasta",
            OptKind.StringLike,
            description="File containing the target sequences.",
        )
    )
    parser.add_opt(
        OptConfig(
            "query-fasta",
            OptKind.StringLike,
            description="File containing the query sequences.",
        )
    )
    parser.add_opt(
        OptConfig(
            "output-file",
            OptKind.StringLike,
            description="File to write the output CSV to.",
        )
    )
    parser.add_opt(
        OptConfig(
            "metrics-file",
            OptKind.StringLike,
            default_value=String("-"),
            description=(
                "File to write the output metrics CSV to. These are the"
                " benchmark results. Writing to - will write to stdout."
            ),
        )
    )
    parser.add_opt(
        OptConfig(
            "iterations",
            OptKind.IntLike,
            default_value=String("1"),
            description="Number of iterations to benchmark.",
        )
    )
    parser.add_opt(
        OptConfig(
            "score-size",
            OptKind.StringLike,
            default_value=String("adaptive"),
            description=(
                "The size to use for scoring:\n\tadaptive: use byte size first"
                " and fall back to word size if it overflows.\n\tbyte:"
                " u8\n\5word: u16"
            ),
        )
    )
    return parser.parse_sys_args()


@value
struct ByteFastaRecord:
    var name: String
    var seq: List[UInt8]
    var rev: List[UInt8]

    fn __init__(
        out self,
        owned name: String,
        owned seq: String,
        read scoring_matrix: ScoringMatrix,
        convert_to_aa: Bool = True,
    ):
        var seq_bytes = List(seq.as_bytes())
        if convert_to_aa:
            seq_bytes = scoring_matrix.convert_ascii_to_encoding(seq_bytes)
        var rev = List[UInt8](capacity=len(seq_bytes))
        for s in reversed(seq_bytes):
            rev.append(s[])
        self.name = name^
        self.seq = seq_bytes^
        self.rev = rev^


@value
struct Profiles[SIMD_U8_WIDTH: Int, SIMD_U16_WIDTH: Int]:
    var fwd: Profile[SIMD_U8_WIDTH, SIMD_U16_WIDTH]
    var rev: Profile[SIMD_U8_WIDTH, SIMD_U16_WIDTH]

    fn __init__(
        out self,
        read record: ByteFastaRecord,
        read matrix: ScoringMatrix,
        score_size: ScoreSize,
    ):
        self.fwd = Profile[SIMD_U8_WIDTH, SIMD_U16_WIDTH](
            record.seq, matrix, score_size
        )
        self.rev = Profile[SIMD_U8_WIDTH, SIMD_U16_WIDTH](
            record.rev, matrix, score_size
        )


@value
@register_passable("trivial")
struct BasicAlignmentOutput(ToDelimited):
    var query_idx: Int
    var target_idx: Int
    var query_len: Int
    var target_len: Int
    var alignment_score: Int
    var query_end: Int
    var target_end: Int

    fn __init__(
        out self,
        query_idx: Int,
        target_idx: Int,
        query_len: Int,
        target_len: Int,
        read result: Alignment,
    ):
        self.query_idx = query_idx
        self.target_idx = target_idx
        self.query_len = query_len
        self.target_len = target_len
        self.alignment_score = Int(result.score1)
        self.query_end = Int(result.read_end1)
        self.target_end = Int(result.ref_end1)

    fn write_to_delimited(read self, mut writer: DelimWriter) raises:
        writer.write_record(
            self.query_idx,
            self.target_idx,
            self.query_len,
            self.target_len,
            self.alignment_score,
            self.query_end,
            self.target_end,
        )

    fn write_header(read self, mut writer: DelimWriter) raises:
        writer.write_record(
            "QueryIdx",
            "TargetIdx",
            "QueryLen",
            "TargetLen",
            "AlignmentScore",
            "QueryEnd",
            "TargetEnd",
        )


@value
struct BenchmarkResults(ToDelimited):
    var total_query_seqs: Int
    var total_target_seqs: Int
    var query_len: Int
    var matrix: String
    var gap_open: Int
    var gap_extend: Int
    var u8_width: Int
    var u16_width: Int
    var score_size: String
    var runtime_secs: Float64
    var cells_updated: UInt64
    var gcups: Float64

    @staticmethod
    fn average(results: List[Self]) raises -> Self:
        var total_query_seqs = 0
        var total_target_seqs = 0
        var query_len = results[0].query_len
        var matrix = results[0].matrix
        var gap_open = results[0].gap_open
        var gap_extend = results[0].gap_extend
        var u8_width = results[0].u8_width
        var u16_width = results[0].u16_width
        var score_size = results[0].score_size
        var runtime_secs = 0.0
        var cells_updated: UInt64 = 0
        var gcups = 0.0

        for result in results:
            total_query_seqs += result[].total_query_seqs
            total_target_seqs += result[].total_target_seqs
            runtime_secs += result[].runtime_secs
            cells_updated += result[].cells_updated
            gcups += result[].gcups

            if query_len != result[].query_len:
                # TODO: may want to change this if we want to process multiple queries in one go.
                raise "Mismatching query len"
            if matrix != result[].matrix:
                raise "Mismatching matrix"
            if gap_open != result[].gap_open:
                raise "Mismatching gap open"
            if gap_extend != result[].gap_extend:
                raise "Mismatching gap extend"
            if u8_width != result[].u8_width:
                raise "Mismatching u8 width"
            if u16_width != result[].u16_width:
                raise "Mismatching u16 width"
            if score_size != result[].score_size:
                raise "Mismatching score size"

        return Self(
            total_query_seqs // len(results),
            total_target_seqs // len(results),
            query_len,
            matrix,
            gap_open,
            gap_extend,
            u8_width,
            u16_width,
            score_size,
            runtime_secs / len(results),
            cells_updated // len(results),
            gcups / len(results),
        )

    fn write_to_delimited(read self, mut writer: DelimWriter) raises:
        writer.write_record(
            self.total_query_seqs,
            self.total_target_seqs,
            self.query_len,
            self.matrix,
            self.gap_open,
            self.gap_extend,
            self.u8_width,
            self.u16_width,
            self.score_size,
            self.runtime_secs,
            self.cells_updated,
            self.gcups,
        )

    fn write_header(read self, mut writer: DelimWriter) raises:
        writer.write_record(
            "total_query_seqs",
            "total_target_seqs",
            "query_len",
            "matrix",
            "gap_open",
            "gap_extend",
            "u8_width",
            "u16_width",
            "score_size",
            "runtime_secs",
            "cells_updated",
            "gcups",
        )


fn main() raises:
    var opts = parse_args()
    if opts.get_bool("help"):
        print(opts.get_help_message()[])
        return None

    var match_score = opts.get_int("match-score")
    var mismatch_score = opts.get_int("mismatch-score")
    var gap_open_score = opts.get_int("gap-open-score")
    var gap_extension_score = opts.get_int("gap-ext-score")
    var output_file = opts.get_string("output-file")
    var metrics_file = opts.get_string("metrics-file")
    var iterations = opts.get_int("iterations")

    # Get the score size
    var score_size_name = opts.get_string("score-size")
    var score_size: ScoreSize
    if score_size_name == "adaptive":
        score_size = ScoreSize.Adaptive
    elif score_size_name == "byte":
        score_size = ScoreSize.Byte
    elif score_size_name == "word":
        score_size = ScoreSize.Word
    else:
        raise "Unkown score size " + score_size_name

    # Create the score matrix
    var matrix_name = opts.get_string("scoring-matrix")
    var matrix: ScoringMatrix
    if matrix_name == "Blosum50":
        matrix = ScoringMatrix.blosum50()
    elif matrix_name == "Blosum62":
        matrix = ScoringMatrix.blosum62()
    elif matrix_name == "ACTGN":
        matrix = ScoringMatrix.actgn_matrix(
            match_score=match_score, mismatch_score=mismatch_score
        )
    else:
        raise "Unknown matrix " + matrix_name

    ## Assuming we are using Blosum50 AA matrix for everything below this for now.

    # Read the fastas and encode the sequences
    var target_file = opts.get_string("target-fasta")
    var target_seqs = FastaRecord.slurp_fasta(target_file)
    var targets = List[ByteFastaRecord](capacity=len(target_seqs))
    while len(target_seqs) > 0:
        var t = target_seqs.pop()
        # I should be able to pass with ^ here, not sure why I can't
        targets.append(ByteFastaRecord(t.name, t.seq, matrix))
    targets.reverse()

    var query_file = opts.get_string("query-fasta")
    var query_seqs = FastaRecord.slurp_fasta(query_file)
    var queries = List[ByteFastaRecord](capacity=len(query_seqs))
    while len(query_seqs) > 0:
        var q = query_seqs.pop()
        queries.append(ByteFastaRecord(q.name, q.seq, matrix))
    queries.reverse()

    # Create query profiles
    var profiles = List[Profiles[SIMD_U8_WIDTH, SIMD_U16_WIDTH]](
        capacity=len(queries)
    )
    for q in queries:
        profiles.append(
            Profiles[SIMD_U8_WIDTH, SIMD_U16_WIDTH](q[], matrix, score_size)
        )

    var results = List[BenchmarkResults]()
    for _ in range(0, iterations):
        var writer = DelimWriter(
            BufferedWriter(open(output_file, "w")),
            write_header=True,
            delim=",",
        )
        # Align
        print("Total query seqs:", len(queries), file=stderr)
        print("Total target seqs:", len(targets), file=stderr)
        print("Using", matrix_name, file=stderr)
        print("Gap open penalty:", gap_open_score, file=stderr)
        print("Gap ext penalty:", gap_extension_score, file=stderr)
        print("U8 SIMD Width:", SIMD_U8_WIDTH, file=stderr)
        print("U16 SIMD Width:", SIMD_U16_WIDTH, file=stderr)
        var start = perf_counter()
        var work: UInt64 = 0
        for i in range(0, len(queries)):
            # TODO: if query construction was done here, we could dispatch to 512 sometimes, which might be cheating for bench purposes.
            var query = Pointer.address_of(queries[i])
            var profiles = Pointer.address_of(profiles[i])
            for j in range(0, len(targets)):
                var target = Pointer.address_of(targets[j])
                var result = ssw_align[SIMD_U8_WIDTH, SIMD_U16_WIDTH](
                    profile=profiles[].fwd,
                    matrix=matrix,
                    reference=target[].seq,
                    query=query[].seq,
                    reverse_profile=profiles[].rev,
                    gap_open_penalty=gap_open_score,
                    gap_extension_penalty=gap_extension_score,
                    return_only_alignment_end=True,
                    mask_length=15,
                )
                if result:
                    writer.serialize(
                        BasicAlignmentOutput(
                            i,
                            j,
                            len(query[].seq),
                            len(target[].seq),
                            result.value(),
                        )
                    )
                else:
                    print("no result")
                work += len(target[].seq) * len(query[].seq)
        var end = perf_counter()

        var elapsed = end - start
        var cells_per_second = work.cast[DType.float64]() / elapsed
        writer.flush()
        print("Ran in", elapsed, "seconds", file=stderr)
        print("Total cells updated :", work, file=stderr)
        print("Cells per second:", cells_per_second, file=stderr)
        print("GCUPs:", cells_per_second / 1000000000, file=stderr)
        results.append(
            BenchmarkResults(
                len(queries),
                len(targets),
                len(queries[0].seq),
                matrix_name,
                gap_open_score,
                gap_extension_score,
                SIMD_U8_WIDTH,
                SIMD_U16_WIDTH,
                String(score_size),
                elapsed,
                work,
                cells_per_second / 1000000000,
            )
        )

    var result = BenchmarkResults.average(results)
    var metric_writer = DelimWriter(
        BufferedWriter(
            open(metrics_file, "w") if metrics_file != "-" else stdout
        ),
        delim=",",
        write_header=True,
    )
    metric_writer.serialize(result)
    metric_writer.flush()
