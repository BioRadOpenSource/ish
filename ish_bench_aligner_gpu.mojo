"""Test program for SSW."""
from sys import stdout, stderr
from sys.info import simdwidthof, has_avx512f, alignof, num_physical_cores
from sys.param_env import env_get_int
from time.time import perf_counter
from utils import StringSlice

from algorithm.functional import parallelize

from ExtraMojo.cli.parser import OptParser, OptConfig, OptKind, ParsedOpts
from ExtraMojo.io.buffered import BufferedWriter
from ExtraMojo.io.delimited import DelimWriter, ToDelimited
from ishlib.matcher.alignment.local_aln.striped import (
    ssw_align,
    Profile,
    ScoreSize,
    Alignment,
)
from ishlib.matcher.alignment.scoring_matrix import (
    ScoringMatrix,
    BasicScoringMatrix,
    BLOSUM62,
    NUM_TO_AA,
)
from ishlib.formats.fasta import FastaRecord
from ishlib.matcher.alignment.striped_utils import AlignmentResult
from ishlib.matcher.alignment.semi_global_aln.striped import (
    semi_global_aln,
    semi_global_aln_with_saturation_check,
    Profile as SemiGlobalProfile,
)
from ishlib.gpu.dynamic_2d_matrix import Dynamic2DMatrix, StorageFormat

from ishlib.matcher.alignment.semi_global_aln.basic import (
    semi_global_parasail,
    semi_global_parasail_gpu,
    SGResult,
)


# GPU
from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim, warp, barrier
from gpu.host import (
    DeviceContext,
    DeviceBuffer,
)  # HostBuffer (not in 25.2)
from gpu.memory import AddressSpace, external_memory
from memory import stack_allocation, memcpy, UnsafePointer
from layout import Layout, LayoutTensor
from math import ceildiv

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

# alias SIMD_U8_WIDTH = simdwidthof[UInt8]() // SIMD_MOD
# alias SIMD_U16_WIDTH = simdwidthof[UInt16]() // SIMD_MOD

alias SIMD_U8_WIDTH = simdwidthof[UInt8]() // 2
alias SIMD_U16_WIDTH = simdwidthof[UInt16]() // 2


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
            "algo",
            OptKind.StringLike,
            default_value=String("striped-local"),
            description=(
                "The alignment algorithm used. [striped-local,"
                " striped-semi-global]"
            ),
        )
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
struct SemiGlobalProfiles[SIMD_U8_WIDTH: Int, SIMD_U16_WIDTH: Int]:
    var fwd: SemiGlobalProfile[SIMD_U8_WIDTH, SIMD_U16_WIDTH]
    var rev: SemiGlobalProfile[SIMD_U8_WIDTH, SIMD_U16_WIDTH]

    fn __init__(
        out self,
        read record: ByteFastaRecord,
        read matrix: ScoringMatrix,
        score_size: ScoreSize,
    ):
        self.fwd = SemiGlobalProfile[SIMD_U8_WIDTH, SIMD_U16_WIDTH](
            record.seq, matrix, score_size
        )
        self.rev = SemiGlobalProfile[SIMD_U8_WIDTH, SIMD_U16_WIDTH](
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

    fn __init__(out self):
        """Create a 'default' object."""
        self.query_idx = 0
        self.target_idx = 0
        self.query_len = 0
        self.target_len = 0
        self.alignment_score = 0
        self.query_end = 0
        self.target_end = 0

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

    fn __init__(
        out self,
        query_idx: Int,
        target_idx: Int,
        query_len: Int,
        target_len: Int,
        read result: AlignmentResult,
    ):
        self.query_idx = query_idx
        self.target_idx = target_idx
        self.query_len = query_len
        self.target_len = target_len
        self.alignment_score = Int(result.best.score)
        self.query_end = Int(result.best.query)
        self.target_end = Int(result.best.reference)

    fn __init__(
        out self,
        query_idx: Int,
        target_idx: Int,
        query_len: Int,
        target_len: Int,
        read result: SGResult,
    ):
        self.query_idx = query_idx
        self.target_idx = target_idx
        self.query_len = query_len
        self.target_len = target_len
        self.alignment_score = Int(result.score)
        self.query_end = Int(result.query)
        self.target_end = Int(result.target)

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

    # Get the algorithm to use
    var algo = opts.get_string("algo")
    var algorithm: String
    if algo == "striped-local":
        algorithm = algo
    elif algo == "striped-semi-global":
        algorithm = algo
    elif algo == "basic-semi-global":
        algorithm = algo
    elif algo == "basic-semi-global-gpu":
        algorithm = algo
    elif algo == "basic-semi-global-gpu-parallel":
        algorithm = algo
    elif algo == "striped-semi-global-parallel":
        algorithm = algo
    else:
        raise "Unknown algo " + algo

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
    elif matrix_name == "ASCII":
        matrix = ScoringMatrix.all_ascii_default_matrix()
    else:
        raise "Unknown matrix " + matrix_name

    var target_file = opts.get_string("target-fasta")
    var query_file = opts.get_string("query-fasta")
    ## Assuming we are using Blosum50 AA matrix for everything below this for now.

    if algorithm == "striped-local":
        bench_striped_local(
            target_file,
            query_file,
            metrics_file,
            output_file,
            matrix,
            score_size,
            iterations,
            matrix_name,
            gap_open_score,
            gap_extension_score,
        )
    elif algorithm == "striped-semi-global":
        bench_striped_semi_global(
            target_file,
            query_file,
            metrics_file,
            output_file,
            matrix,
            score_size,
            iterations,
            matrix_name,
            gap_open_score,
            gap_extension_score,
        )
    elif algorithm == "basic-semi-global":
        bench_basic_semi_global(
            target_file,
            query_file,
            metrics_file,
            output_file,
            matrix,
            score_size,
            iterations,
            matrix_name,
            gap_open_score,
            gap_extension_score,
        )
    elif algorithm == "basic-semi-global-gpu":
        bench_basic_semi_global_gpu(
            target_file,
            query_file,
            metrics_file,
            output_file,
            matrix,
            score_size,
            iterations,
            matrix_name,
            gap_open_score,
            gap_extension_score,
        )
    elif algorithm == "basic-semi-global-gpu-parallel":
        bench_basic_semi_global_gpu_parallel(
            target_file,
            query_file,
            metrics_file,
            output_file,
            matrix,
            score_size,
            iterations,
            matrix_name,
            gap_open_score,
            gap_extension_score,
        )
    elif algorithm == "striped-semi-global-parallel":
        bench_striped_semi_global_parallel(
            target_file,
            query_file,
            metrics_file,
            output_file,
            matrix,
            score_size,
            iterations,
            matrix_name,
            gap_open_score,
            gap_extension_score,
        )
    else:
        raise "Unknown algo " + algorithm


fn bench_striped_local(
    target_file: String,
    query_file: String,
    metrics_file: String,
    output_file: String,
    matrix: ScoringMatrix,
    score_size: ScoreSize,
    iterations: Int,
    matrix_name: String,
    gap_open_score: Int,
    gap_extension_score: Int,
) raises:
    var prep_start = perf_counter()
    # Read the fastas and encode the sequences
    var target_seqs = FastaRecord.slurp_fasta(target_file)
    var targets = List[ByteFastaRecord](capacity=len(target_seqs))
    while len(target_seqs) > 0:
        var t = target_seqs.pop()
        # I should be able to pass with ^ here, not sure why I can't
        targets.append(ByteFastaRecord(t.name, t.seq, matrix))
    targets.reverse()

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
    var prep_end = perf_counter()
    print("Setup Time:", prep_end - prep_start, file=stderr)

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


fn bench_striped_semi_global(
    target_file: String,
    query_file: String,
    metrics_file: String,
    output_file: String,
    matrix: ScoringMatrix,
    score_size: ScoreSize,
    iterations: Int,
    matrix_name: String,
    gap_open_score: Int,
    gap_extension_score: Int,
) raises:
    # Read the fastas and encode the sequences
    var prep_start = perf_counter()
    var target_seqs = FastaRecord.slurp_fasta(target_file)

    fn cmp_lens(lhs: FastaRecord, rhs: FastaRecord) capturing -> Bool:
        return len(lhs.seq) > len(rhs.seq)

    # TODO: be better
    # Try to group like-lengths together
    sort[cmp_lens](target_seqs)

    var targets = List[ByteFastaRecord](capacity=len(target_seqs))
    while len(target_seqs) > 0:
        var t = target_seqs.pop()
        # if len(t.seq) > 1024:
        #     continue
        # I should be able to pass with ^ here, not sure why I can't
        targets.append(ByteFastaRecord(t.name, t.seq, matrix))
    targets.reverse()

    var query_seqs = FastaRecord.slurp_fasta(query_file)
    var queries = List[ByteFastaRecord](capacity=len(query_seqs))
    while len(query_seqs) > 0:
        var q = query_seqs.pop()
        queries.append(ByteFastaRecord(q.name, q.seq, matrix))
    queries.reverse()

    # Create query profiles
    var profiles = List[SemiGlobalProfiles[SIMD_U8_WIDTH, SIMD_U16_WIDTH]](
        capacity=len(queries)
    )
    for q in queries:
        profiles.append(
            SemiGlobalProfiles[SIMD_U8_WIDTH, SIMD_U16_WIDTH](
                q[], matrix, score_size
            )
        )
    var prep_end = perf_counter()
    print("Setup Time:", prep_end - prep_start, file=stderr)

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
        print("Bias:", profiles[0].fwd.bias, file=stderr)
        var start = perf_counter()
        var work: UInt64 = 0
        for i in range(0, len(queries)):
            # TODO: if query construction was done here, we could dispatch to 512 sometimes, which might be cheating for bench purposes.
            var query = Pointer.address_of(queries[i])
            var profiles = Pointer.address_of(profiles[i])
            for j in range(0, len(targets)):
                var target = Pointer.address_of(targets[j])
                var result = semi_global_aln_with_saturation_check[
                    SIMD_U8_WIDTH,
                    SIMD_U16_WIDTH,
                ](
                    profile=profiles[].fwd,
                    reference=target[].seq,
                    query_len=len(query[].seq),
                    gap_open_penalty=gap_open_score,
                    gap_extension_penalty=gap_extension_score,
                    free_query_start_gaps=True,
                    free_query_end_gaps=True,
                    free_target_start_gaps=True,
                    free_target_end_gaps=True,
                    score_size=score_size,
                )
                writer.serialize(
                    BasicAlignmentOutput(
                        i,
                        j,
                        len(query[].seq),
                        len(target[].seq),
                        result,
                    )
                )
                work += len(target[].seq) * len(query[].seq)
        var end = perf_counter()

        var elapsed = end - start
        var cells_per_second = work.cast[DType.float64]() / elapsed
        writer.flush()
        print("Algo: striped-semi-global", file=stderr)
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


fn bench_basic_semi_global(
    target_file: String,
    query_file: String,
    metrics_file: String,
    output_file: String,
    matrix: ScoringMatrix,
    score_size: ScoreSize,
    iterations: Int,
    matrix_name: String,
    gap_open_score: Int,
    gap_extension_score: Int,
) raises:
    # Read the fastas and encode the sequences
    var prep_start = perf_counter()
    var target_seqs = FastaRecord.slurp_fasta(target_file)

    fn cmp_lens(lhs: FastaRecord, rhs: FastaRecord) capturing -> Bool:
        return len(lhs.seq) > len(rhs.seq)

    # TODO: be better
    # Try to group like-lengths together
    sort[cmp_lens](target_seqs)

    var targets = List[ByteFastaRecord](capacity=len(target_seqs))
    while len(target_seqs) > 0:
        var t = target_seqs.pop()
        if len(t.seq) > 1024:  # or len(t.seq) < 100:
            continue
        # if (
        #     t.name
        #     != "sp|Q99VY7|MNHF2_STAAM Putative antiporter subunit mnhF2"
        #     " OS=Staphylococcus aureus (strain Mu50 / ATCC 700699) GN=mnhF2"
        #     " PE=3 SV=1"
        # ):
        #     continue
        # print(t.name)
        # I should be able to pass with ^ here, not sure why I can't
        targets.append(ByteFastaRecord(t.name, t.seq, matrix))
        # print("Remove break here")
        # break
    targets.reverse()

    var query_seqs = FastaRecord.slurp_fasta(query_file)
    var queries = List[ByteFastaRecord](capacity=len(query_seqs))
    while len(query_seqs) > 0:
        var q = query_seqs.pop()
        queries.append(ByteFastaRecord(q.name, q.seq, matrix))
    queries.reverse()

    # TODO: currently leaking this memory
    var basic_matrix_ptr = UnsafePointer[Int8].alloc(len(BLOSUM62))
    memcpy(basic_matrix_ptr, BLOSUM62.unsafe_ptr(), len(BLOSUM62))
    var basic_matrix = BasicScoringMatrix(basic_matrix_ptr, len(BLOSUM62))
    var truth = ScoringMatrix.blosum62()
    print("Truth, basic", truth.size, basic_matrix.size)
    for i in range(0, len(truth.values)):
        if truth.values[i] != basic_matrix.values[i]:
            print("Not EQUAL")

    # Create query profiles
    var prep_end = perf_counter()
    print("Setup Time:", prep_end - prep_start, file=stderr)

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
            for j in range(0, len(targets)):
                var target = Pointer.address_of(targets[j])

                var result = semi_global_parasail[DType.int16](
                    query[].seq,
                    target[].seq,
                    # matrix,
                    basic_matrix,
                    gap_open_penalty=-gap_open_score,
                    gap_extension_penalty=-gap_extension_score,
                    free_query_start_gaps=True,
                    free_query_end_gaps=True,
                    free_target_start_gaps=True,
                    free_target_end_gaps=True,
                )

                writer.serialize(
                    BasicAlignmentOutput(
                        i,
                        j,
                        len(query[].seq),
                        len(target[].seq),
                        result,
                    )
                )
                work += len(target[].seq) * len(query[].seq)
        var end = perf_counter()

        var elapsed = end - start
        var cells_per_second = work.cast[DType.float64]() / elapsed
        writer.flush()
        print("Algo: basic-semi-global", file=stderr)
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


fn bench_striped_semi_global_parallel(
    target_file: String,
    query_file: String,
    metrics_file: String,
    output_file: String,
    matrix: ScoringMatrix,
    score_size: ScoreSize,
    iterations: Int,
    matrix_name: String,
    gap_open_score: Int,
    gap_extension_score: Int,
) raises:
    # Read the fastas and encode the sequences
    var prep_start = perf_counter()
    var target_seqs = FastaRecord.slurp_fasta(target_file)

    fn cmp_lens(lhs: FastaRecord, rhs: FastaRecord) capturing -> Bool:
        return len(lhs.seq) > len(rhs.seq)

    # TODO: be better
    # Try to group like-lengths together
    sort[cmp_lens](target_seqs)

    var targets = List[ByteFastaRecord](capacity=len(target_seqs))
    while len(target_seqs) > 0:
        var t = target_seqs.pop()
        if len(t.seq) > 1024:
            continue
        # I should be able to pass with ^ here, not sure why I can't
        targets.append(ByteFastaRecord(t.name, t.seq, matrix))
    targets.reverse()

    var query_seqs = FastaRecord.slurp_fasta(query_file)
    var queries = List[ByteFastaRecord](capacity=len(query_seqs))
    while len(query_seqs) > 0:
        var q = query_seqs.pop()
        queries.append(ByteFastaRecord(q.name, q.seq, matrix))
    queries.reverse()

    var work: UInt64 = 0
    # for target in targets:
    #     work += len(queries[0].seq) * len(target[].seq)

    # Create query profiles
    var profiles = List[SemiGlobalProfiles[SIMD_U8_WIDTH, SIMD_U16_WIDTH]](
        capacity=len(queries)
    )
    for q in queries:
        profiles.append(
            SemiGlobalProfiles[SIMD_U8_WIDTH, SIMD_U16_WIDTH](
                q[], matrix, score_size
            )
        )
    var output = List[BasicAlignmentOutput](capacity=len(targets))
    for _ in range(0, len(targets)):
        output.append(BasicAlignmentOutput())

    var prep_end = perf_counter()
    print("Setup Time:", prep_end - prep_start, file=stderr)

    var results = List[BenchmarkResults]()
    for _ in range(0, iterations):
        var writer = DelimWriter(
            BufferedWriter(open(output_file, "w")),
            write_header=True,
            delim=",",
        )
        # Align
        print("Pysical Cores:", num_physical_cores() * 2, file=stderr)
        print("Total query seqs:", len(queries), file=stderr)
        print("Total target seqs:", len(targets), file=stderr)
        print("Using", matrix_name, file=stderr)
        print("Gap open penalty:", gap_open_score, file=stderr)
        print("Gap ext penalty:", gap_extension_score, file=stderr)
        print("U8 SIMD Width:", SIMD_U8_WIDTH, file=stderr)
        print("U16 SIMD Width:", SIMD_U16_WIDTH, file=stderr)
        print("Bias:", profiles[0].fwd.bias, file=stderr)
        var start = perf_counter()
        # var work: UInt64 = 0
        for i in range(0, len(queries)):
            # TODO: if query construction was done here, we could dispatch to 512 sometimes, which might be cheating for bench purposes.
            var query = Pointer.address_of(queries[i])
            var profiles = Pointer.address_of(profiles[i])

            fn do_alignment(index: Int) capturing:
                var target = Pointer.address_of(targets[index])
                if index == 532959:
                    print("532959 is:", targets[index].name)
                var result = semi_global_aln_with_saturation_check[
                    SIMD_U8_WIDTH,
                    SIMD_U16_WIDTH,
                ](
                    profile=profiles[].fwd,
                    reference=target[].seq,
                    query_len=len(query[].seq),
                    gap_open_penalty=gap_open_score,
                    gap_extension_penalty=gap_extension_score,
                    free_query_start_gaps=True,
                    free_query_end_gaps=True,
                    free_target_start_gaps=True,
                    free_target_end_gaps=True,
                    score_size=score_size,
                )
                output[index] = BasicAlignmentOutput(
                    i,
                    index,
                    len(query[].seq),
                    len(target[].seq),
                    result,
                )

            parallelize[do_alignment](len(targets), num_physical_cores())

            for j in range(0, len(targets)):
                var target = Pointer.address_of(targets[j])
                writer.serialize(output[j])
                work += len(target[].seq) * len(query[].seq)
        var end = perf_counter()

        var elapsed = end - start
        var cells_per_second = work.cast[DType.float64]() / elapsed
        writer.flush()
        print("Algo: striped-semi-global", file=stderr)
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


fn bench_basic_semi_global_gpu(
    target_file: String,
    query_file: String,
    metrics_file: String,
    output_file: String,
    matrix: ScoringMatrix,
    score_size: ScoreSize,
    iterations: Int,
    matrix_name: String,
    gap_open_score: Int,
    gap_extension_score: Int,
) raises:
    alias NUM_SEQS = 533591
    ################
    # Setup
    ################
    # Read the fastas and encode the sequences
    var prep_start = perf_counter()
    var target_seqs = FastaRecord.slurp_fasta(target_file)

    # TODO: currently leaking this memory
    var basic_matrix_ptr = UnsafePointer[Int8].alloc(len(BLOSUM62))
    memcpy(basic_matrix_ptr, BLOSUM62.unsafe_ptr(), len(BLOSUM62))
    var basic_matrix = BasicScoringMatrix(basic_matrix_ptr, len(BLOSUM62))

    fn cmp_lens(lhs: FastaRecord, rhs: FastaRecord) capturing -> Bool:
        return len(lhs.seq) > len(rhs.seq)

    # TODO: be better
    # Try to group like-lengths together
    sort[cmp_lens](target_seqs)
    target_seqs.reverse()

    var targets = List[ByteFastaRecord](capacity=len(target_seqs))
    var target_ends = List[Int]()
    while len(target_seqs) > 0:
        var t = target_seqs.pop()
        if len(t.seq) > 1024:  # or len(t.seq) < 1023:
            continue
        # if (
        #     t.name
        #     != "sp|Q99VY7|MNHF2_STAAM Putative antiporter subunit mnhF2"
        #     " OS=Staphylococcus aureus (strain Mu50 / ATCC 700699) GN=mnhF2"
        #     " PE=3 SV=1"
        # ):
        #     continue
        # print(t.name)
        # I should be able to pass with ^ here, not sure why I can't
        var r = ByteFastaRecord(t.name, t.seq, matrix)
        target_ends.append(len(t.seq))
        # print("start, end, seq: ", start, end, t.seq)
        targets.append(r^)
        # print("Remove break here")
        # break
    # targets.reverse()
    # ref_coords.reverse()

    var query_seqs = FastaRecord.slurp_fasta(query_file)
    var queries = List[ByteFastaRecord](capacity=len(query_seqs))
    while len(query_seqs) > 0:
        var q = query_seqs.pop()
        queries.append(ByteFastaRecord(q.name, q.seq, matrix))
    queries.reverse()

    # Create query profiles
    var prep_end = perf_counter()
    print("Setup Time:", prep_end - prep_start, file=stderr)
    var writer = DelimWriter(
        BufferedWriter(open(output_file, "w")),
        write_header=True,
        delim=",",
    )

    ################
    # Copy to GPU
    ################

    var ctx = DeviceContext()

    var host_ref_buffer = ctx.enqueue_create_host_buffer[DType.uint8](
        # ref_total_bytes
        NUM_SEQS
        * 1024
    )
    var host_target_ends = ctx.enqueue_create_host_buffer[DType.uint32](
        len(target_ends)
    )
    ctx.synchronize()
    # var host_ref_buffer_ptr = host_ref_buffer.unsafe_ptr()
    # var byte_count = 0
    # var elem_count = 0

    var tensor = LayoutTensor[DType.uint8, Layout.col_major(1024, NUM_SEQS)](
        host_ref_buffer
    )

    print("Total targets: ", len(targets))
    for i in range(0, len(targets)):
        for j in range(0, len(targets[i].seq)):
            tensor[j, i] = targets[i].seq[j]
        host_target_ends[i] = len(targets[i].seq)

    # No other methods count their data prep time for GCUPS,
    # ADEPT doesn't even seem to count data transfer time, but cudaSW4 does, so we will too.
    # This does miss out on the host allocation time, but that is negligable
    # and we could create the array elswere.
    var gpu_start = perf_counter()
    var dev_ref_buffer = ctx.enqueue_create_buffer[DType.uint8](NUM_SEQS * 1024)
    var dev_target_ends = ctx.enqueue_create_buffer[DType.uint32](
        len(target_ends)
    )
    host_ref_buffer.enqueue_copy_to(dev_ref_buffer)
    host_target_ends.enqueue_copy_to(dev_target_ends)

    var work = 0
    # Create the host buffers
    for i in range(0, len(queries)):
        var query = Pointer.address_of(queries[i])

        var host_query = ctx.enqueue_create_host_buffer[DType.uint8](
            len(query[].seq)
        )
        var host_basic_matrix = ctx.enqueue_create_host_buffer[DType.int8](
            len(basic_matrix)
        )

        var dev_query = ctx.enqueue_create_buffer[DType.uint8](len(query[].seq))
        var dev_basic_matrix = ctx.enqueue_create_buffer[DType.int8](
            len(basic_matrix)
        )

        # Result buffers
        var host_score_result_buffer = ctx.enqueue_create_host_buffer[
            DType.int32
        ](NUM_SEQS)
        var host_query_end_result_buffer = ctx.enqueue_create_host_buffer[
            DType.int32
        ](NUM_SEQS)
        var host_ref_end_result_buffer = ctx.enqueue_create_host_buffer[
            DType.int32
        ](NUM_SEQS)

        var dev_score_result_buffer = ctx.enqueue_create_buffer[DType.int32](
            NUM_SEQS
        )
        var dev_query_end_result_buffer = ctx.enqueue_create_buffer[
            DType.int32
        ](NUM_SEQS)
        var dev_ref_end_result_buffer = ctx.enqueue_create_buffer[DType.int32](
            NUM_SEQS
        )

        ctx.enqueue_memset(dev_score_result_buffer, 0)
        ctx.enqueue_memset(dev_query_end_result_buffer, 0)
        ctx.enqueue_memset(dev_ref_end_result_buffer, 0)

        memcpy(
            host_query.unsafe_ptr(), query[].seq.unsafe_ptr(), len(query[].seq)
        )
        memcpy(
            host_basic_matrix.unsafe_ptr(),
            basic_matrix.values,
            len(basic_matrix),
        )

        host_query.enqueue_copy_to(dev_query)
        host_basic_matrix.enqueue_copy_to(dev_basic_matrix)

        ctx.synchronize()

        # Use batched processing instead of single kernel launch
        # process_in_batches(
        #     ctx,
        #     dev_query,
        #     dev_ref_buffer,
        #     dev_target_ends,
        #     dev_score_result_buffer,
        #     dev_query_end_result_buffer,
        #     dev_ref_end_result_buffer,
        #     dev_basic_matrix,
        #     len(basic_matrix),
        #     len(query[].seq),
        #     len(target_ends),
        #     # batch_size=1,
        #     batch_size=15000,  # L4
        #     # batch_size=45000,  # L40S
        #     # batch_size=1,
        # )
        # print("Done with batches")
        process_with_coarse_graining(
            ctx,
            dev_query,
            dev_ref_buffer,
            dev_target_ends,
            dev_score_result_buffer,
            dev_query_end_result_buffer,
            dev_ref_end_result_buffer,
            dev_basic_matrix,
            len(basic_matrix),
            len(query[].seq),
            len(target_ends),
            threads_to_launch=15000,
        )

        # Copy results back to host
        dev_score_result_buffer.enqueue_copy_to(host_score_result_buffer)
        dev_query_end_result_buffer.enqueue_copy_to(
            host_query_end_result_buffer
        )
        dev_ref_end_result_buffer.enqueue_copy_to(host_ref_end_result_buffer)
        ctx.synchronize()

        # Write results
        for j in range(0, len(target_ends)):
            work += len(query[].seq) * len(targets[j].seq)
            writer.serialize(
                BasicAlignmentOutput(
                    i,
                    j,
                    len(query[].seq),
                    len(targets[j].seq),
                    Int(host_score_result_buffer[j]),
                    Int(host_query_end_result_buffer[j]),
                    Int(host_ref_end_result_buffer[j]),
                )
            )

    var gpu_end = perf_counter()
    writer.flush()
    var elapsed = gpu_end - gpu_start
    var cells_per_second = work / elapsed
    print("GPU TOOK:", elapsed)
    print("Algo: basic-semi-global-gpu", file=stderr)
    print("Ran in", elapsed, "seconds", file=stderr)
    print("Total cells updated :", work, file=stderr)
    print("Cells per second:", cells_per_second, file=stderr)
    print("GCUPs:", cells_per_second / 1000000000, file=stderr)
    var results = List[BenchmarkResults]()
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


@value
struct BufferPair[dtype: DType]:
    # TODO: maybe we don't want to pair these up, otherwise they can't be destroyed easily?
    var host: DeviceBuffer[dtype]
    var device: DeviceBuffer[dtype]
    var length: UInt

    fn __init__(out self, read ctx: DeviceContext, length: UInt) raises:
        self.host = ctx.enqueue_create_host_buffer[dtype](length)
        self.device = ctx.enqueue_create_buffer[dtype](length)
        self.length = length

    fn __init__(
        out self,
        read ctx: DeviceContext,
        length: UInt,
        device_memset_value: Scalar[dtype],
    ) raises:
        self.host = ctx.enqueue_create_host_buffer[dtype](length)
        self.device = ctx.enqueue_create_buffer[dtype](length)
        self.length = length
        ctx.enqueue_memset(self.device, device_memset_value)

    fn copy_to_device(read self) raises:
        self.host.enqueue_copy_to(self.device)

    fn copy_to_host(read self) raises:
        self.device.enqueue_copy_to(self.host)


@value
struct AlignerDevice:
    var ctx: DeviceContext
    var query: Optional[BufferPair[DType.uint8]]
    var targets: Optional[BufferPair[DType.uint8]]
    var target_lengths: Optional[BufferPair[DType.uint32]]
    var scoring_matrix: Optional[BufferPair[DType.int8]]
    var scores: Optional[BufferPair[DType.int32]]
    var query_ends: Optional[BufferPair[DType.int32]]
    var target_ends: Optional[BufferPair[DType.int32]]

    fn __init__(out self, owned ctx: DeviceContext):
        self.ctx = ctx^
        self.query = None
        self.targets = None
        self.target_lengths = None
        self.scoring_matrix = None
        self.scores = None
        self.query_ends = None
        self.target_ends = None

    fn create_buffers(
        mut self,
        num_targets: UInt,
        query_len: UInt,
        matrix_len: UInt,
        max_target_length: UInt = 1024,
    ) raises:
        self.query = BufferPair[DType.uint8](self.ctx, query_len)
        self.targets = BufferPair[DType.uint8](
            self.ctx, num_targets * max_target_length
        )
        print("Creating buffers for", num_targets, "targets")
        self.target_lengths = BufferPair[DType.uint32](self.ctx, num_targets)
        self.scoring_matrix = BufferPair[DType.int8](self.ctx, matrix_len)
        self.scores = BufferPair[DType.int32](self.ctx, num_targets, 0)
        self.query_ends = BufferPair[DType.int32](self.ctx, num_targets, 0)
        self.target_ends = BufferPair[DType.int32](self.ctx, num_targets, 0)

    fn synchronize(read self) raises:
        self.ctx.synchronize()

    fn set_host_inputs(
        mut self,
        query: Span[UInt8],
        read score_matrix: UnsafePointer[Int8],
        score_matrix_len: UInt,
        targets: Span[ByteFastaRecord],
        max_target_length: UInt = 1024,
    ):
        memcpy(
            self.query.value().host.unsafe_ptr(), query.unsafe_ptr(), len(query)
        )

        memcpy(
            self.scoring_matrix.value().host.unsafe_ptr(),
            score_matrix,
            score_matrix_len,
        )
        # Create interleaved target seqs
        var buffer = self.targets.value().host
        var coords = Dynamic2DMatrix[StorageFormat.ColumnMajor](
            max_target_length, len(targets)
        )

        for i in range(0, len(targets)):
            for j in range(0, len(targets[i].seq)):
                buffer[coords.cord2idx(j, i)] = targets[i].seq[j]
            self.target_lengths.value().host[i] = len(targets[i].seq)

    fn copy_inputs_to_device(read self) raises:
        self.targets.value().copy_to_device()
        self.target_lengths.value().copy_to_device()
        self.query.value().copy_to_device()
        self.scoring_matrix.value().copy_to_device()

    fn copy_outputs_to_host(read self) raises:
        self.scores.value().copy_to_host()
        self.query_ends.value().copy_to_host()
        self.target_ends.value().copy_to_host()

    fn launch_kernel(mut self, threads_to_launch: UInt = 15000) raises:
        process_with_coarse_graining(
            self.ctx,
            self.query.value().device,
            self.targets.value().device,
            self.target_lengths.value().device,
            self.scores.value().device,
            self.query_ends.value().device,
            self.target_ends.value().device,
            self.scoring_matrix.value().device,
            self.scoring_matrix.value().length,
            self.query.value().length,
            self.target_lengths.value().length,
            threads_to_launch=threads_to_launch,
        )

    fn results(read self, idx: UInt) -> (Int32, Int32, Int32):
        return (
            self.scores.value().host[idx],
            self.query_ends.value().host[idx],
            self.target_ends.value().host[idx],
        )


fn bench_basic_semi_global_gpu_parallel(
    target_file: String,
    query_file: String,
    metrics_file: String,
    output_file: String,
    matrix: ScoringMatrix,
    score_size: ScoreSize,
    iterations: Int,
    matrix_name: String,
    gap_open_score: Int,
    gap_extension_score: Int,
) raises:
    alias NUM_SEQS = 533591
    ################
    # Setup
    ################
    # Read the fastas and encode the sequences
    var prep_start = perf_counter()
    var target_seqs = FastaRecord.slurp_fasta(target_file)

    # TODO: currently leaking this memory
    var basic_matrix_ptr = UnsafePointer[Int8].alloc(len(BLOSUM62))
    memcpy(basic_matrix_ptr, BLOSUM62.unsafe_ptr(), len(BLOSUM62))
    var basic_matrix = BasicScoringMatrix(basic_matrix_ptr, len(BLOSUM62))

    fn cmp_lens(lhs: FastaRecord, rhs: FastaRecord) capturing -> Bool:
        return len(lhs.seq) > len(rhs.seq)

    # TODO: be better
    # Try to group like-lengths together
    sort[cmp_lens](target_seqs)
    target_seqs.reverse()

    var targets = List[ByteFastaRecord](capacity=len(target_seqs))
    var target_ends = List[Int]()
    while len(target_seqs) > 0:
        var t = target_seqs.pop()
        if len(t.seq) > 1024:  # or len(t.seq) < 1023:
            continue
        # if (
        #     t.name
        #     != "sp|Q99VY7|MNHF2_STAAM Putative antiporter subunit mnhF2"
        #     " OS=Staphylococcus aureus (strain Mu50 / ATCC 700699) GN=mnhF2"
        #     " PE=3 SV=1"
        # ):
        #     continue
        # print(t.name)
        # I should be able to pass with ^ here, not sure why I can't
        var r = ByteFastaRecord(t.name, t.seq, matrix)
        target_ends.append(len(t.seq))
        # print("start, end, seq: ", start, end, t.seq)
        targets.append(r^)
        # print("Remove break here")
        # break
    # targets.reverse()
    # ref_coords.reverse()

    var query_seqs = FastaRecord.slurp_fasta(query_file)
    var queries = List[ByteFastaRecord](capacity=len(query_seqs))
    while len(query_seqs) > 0:
        var q = query_seqs.pop()
        queries.append(ByteFastaRecord(q.name, q.seq, matrix))
    queries.reverse()

    # Create query profiles
    var prep_end = perf_counter()
    print("Setup Time:", prep_end - prep_start, file=stderr)
    var writer = DelimWriter(
        BufferedWriter(open(output_file, "w")),
        write_header=True,
        delim=",",
    )

    ################
    # Copy to GPU
    ################
    var num_devices = DeviceContext.number_of_devices()
    num_devices = 1
    var ctxs = List[AlignerDevice](capacity=num_devices)
    for i in range(0, num_devices):
        ctxs.append(AlignerDevice(DeviceContext(i)))

    var targets_per_device = ceildiv(len(target_ends), num_devices)
    print("Devices:", num_devices)
    print("Total targets:", len(target_ends))
    print("Targets per device:", targets_per_device)

    # Create Buffers
    for ctx_id in range(0, num_devices):
        ctxs[ctx_id].create_buffers(
            targets_per_device,
            len(queries[0].seq),
            len(basic_matrix),
            max_target_length=1024,
        )

    for ctx_id in range(0, num_devices):
        ctxs[ctx_id].synchronize()

    # Fill in input data
    var device_id = 0
    for i in range(0, len(target_ends), targets_per_device):
        ctxs[device_id].set_host_inputs(
            Span(queries[0].seq),
            basic_matrix.values,
            len(basic_matrix),
            targets[i : min(i + targets_per_device, len(targets))],
            max_target_length=1024,
        )
        ctxs[device_id].copy_inputs_to_device()
        device_id += 1

    # Don't count the data setup time, but do count the transfer time
    var gpu_start = perf_counter()
    for ctx_id in range(0, num_devices):
        ctxs[ctx_id].synchronize()
    var copy_up_done = perf_counter()
    var copy_up_time = copy_up_done - gpu_start
    print("Copy up time:", copy_up_done - gpu_start)

    # Launch kernels
    var kernel_start_time = perf_counter()
    for ctx_id in range(0, num_devices):
        ctxs[ctx_id].launch_kernel(threads_to_launch=15000)

    for ctx_id in range(0, num_devices):
        ctxs[ctx_id].synchronize()
        ctxs[ctx_id].copy_outputs_to_host()
    var kernel_done = perf_counter()
    var kernel_time = kernel_done - kernel_start_time
    print("Kenrel runtime only:", kernel_done - kernel_start_time)

    for ctx_id in range(0, num_devices):
        ctxs[ctx_id].synchronize()
    var copy_down_done = perf_counter()
    var copy_down_time = copy_down_done - kernel_done
    print("Copy down time:", copy_down_done - kernel_done)

    # Write output
    var work = 0

    var write_start = perf_counter()
    var items = 0
    for ctx_id in range(0, num_devices):
        # Write results
        for j in range(0, ctxs[ctx_id].target_lengths.value().length):
            if items >= len(targets):
                break
            var query_len = len(queries[0].seq)
            var target_len = ctxs[ctx_id].target_lengths.value().host[j]
            (score, query_end, target_end) = ctxs[ctx_id].results(j)
            work += Int(query_len) * Int(target_len)
            writer.serialize(
                BasicAlignmentOutput(
                    0,
                    j,
                    query_len,
                    Int(target_len),
                    Int(score),
                    Int(query_end),
                    Int(target_end),
                )
            )
            items += 1

    var gpu_end = perf_counter()
    print("Write time:", gpu_end - write_start)
    writer.flush()
    var elapsed = gpu_end - gpu_start
    var cells_per_second = work / elapsed
    print("Num Devices:", num_devices, file=stderr)
    print("GPU TOOK:", elapsed, file=stderr)
    print("Algo: basic-semi-global-gpu", file=stderr)
    print("Ran in", elapsed, "seconds", file=stderr)
    print("Total cells updated :", work, file=stderr)
    print("Cells per second:", cells_per_second, file=stderr)
    print("GCUPs:", cells_per_second / 1000000000, file=stderr)
    print(
        "gpu GCUPs:",
        (work / (kernel_time + copy_up_time + copy_down_time) / 1000000000),
    )
    var results = List[BenchmarkResults]()
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


##############################################
# GPU Functionality
##############################################


@value
@register_passable("trivial")
struct LineCoords:
    alias UPPER32 = UInt32.MAX.cast[DType.uint64]() << 32
    var start: UInt32
    """0-based inclusive start."""
    var end: UInt32
    """Exclusive end."""

    fn to_u64(self) -> UInt64:
        var value: UInt64 = 0
        value += self.start.cast[DType.uint64]()
        value <<= 32
        value += self.end.cast[DType.uint64]()
        return value

    @staticmethod
    fn from_u64(value: UInt64) -> Self:
        var start = value >> 32
        var end = value ^ Self.UPPER32
        return Self(start.cast[DType.uint32](), end.cast[DType.uint32]())


# Add this function to process references in batches
fn process_in_batches(
    ctx: DeviceContext,
    read dev_query: DeviceBuffer[DType.uint8],
    read dev_ref_buffer: DeviceBuffer[DType.uint8],
    read dev_target_ends: DeviceBuffer[DType.uint32],
    mut dev_score_result_buffer: DeviceBuffer[DType.int32],
    mut dev_query_end_result_buffer: DeviceBuffer[DType.int32],
    mut dev_ref_end_result_buffer: DeviceBuffer[DType.int32],
    read dev_basic_matrix: DeviceBuffer[DType.int8],
    basic_matrix_len: Int,
    query_len: Int,
    target_ends_len: Int,
    batch_size: Int = 6000,
) raises:
    alias BLOCK_SIZE = 32  # Align with warp size
    # alias BLOCK_SIZE = 1

    var num_batches = ceildiv(target_ends_len, batch_size)
    print(
        "Processing",
        target_ends_len,
        "references in",
        num_batches,
        "batches of size",
        batch_size,
    )

    var aligner = ctx.compile_function[gpu_align_batched]()

    for batch in range(num_batches):
        var start_idx = batch * batch_size
        var end_idx = min(start_idx + batch_size, target_ends_len)
        var batch_size_actual = end_idx - start_idx

        print(
            "Processing batch",
            batch + 1,
            "of",
            num_batches,
            "- indices",
            start_idx,
            "to",
            end_idx - 1,
        )

        # Launch kernel for this batch only
        # TODO: precompile
        ctx.enqueue_function(
            aligner,
            dev_query,
            dev_ref_buffer,
            dev_target_ends,
            dev_score_result_buffer,
            dev_query_end_result_buffer,
            dev_ref_end_result_buffer,
            dev_basic_matrix,
            basic_matrix_len,
            query_len,
            target_ends_len,
            start_idx,  # Pass the start index
            batch_size_actual,  # Pass the actual batch size
            grid_dim=ceildiv(batch_size_actual, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )

        # Synchronize after each batch to ensure it completes
        # ctx.synchronize()


# Modified kernel that processes only indices within a specific batch
fn gpu_align_batched(
    query: DeviceBuffer[DType.uint8],
    ref_buffer: DeviceBuffer[DType.uint8],
    target_ends: DeviceBuffer[DType.uint32],
    score_result_buffer: DeviceBuffer[DType.int32],
    query_end_result_buffer: DeviceBuffer[DType.int32],
    ref_end_result_buffer: DeviceBuffer[DType.int32],
    basic_matrix_values: DeviceBuffer[DType.int8],
    basic_matrix_len: Int,
    query_len: Int,
    target_ends_len: Int,
    batch_start_idx: Int,  # Start index for this batch
    batch_size: Int,  # Size of this batch
):
    var basic_profile_bytes = stack_allocation[
        576,
        SIMD[DType.int8, 1],
        address_space = AddressSpace.SHARED,
        # alignment = alignof[Int8](),
    ]()
    # cp_async_bulk_tensor_shared_cluster_global_multicast
    for i in range(0, 576):
        basic_profile_bytes[i] = basic_matrix_values[i]
    var basic_matrix = BasicScoringMatrix[address_space = AddressSpace(3)](
        basic_profile_bytes, 576
    )

    # Query Seq
    var query_seq_ptr = stack_allocation[
        200,
        SIMD[DType.uint8, 1],
        address_space = AddressSpace.SHARED,
        # alignment = alignof[UInt8](),
        # alignment=128,
    ]()
    for i in range(0, query_len):
        query_seq_ptr[i] = query[i]

    barrier()

    # Calculate global thread index
    var local_idx = (block_idx.x * block_dim.x) + thread_idx.x

    # Convert to global index
    var idx = batch_start_idx + local_idx

    # Skip if this thread is outside the batch range
    if local_idx >= batch_size or idx >= target_ends_len:
        return

    # Hard coded for now:
    var gap_open_penalty = -3
    var gap_ext_penalty = -1

    # Get the start/end coords for this ref seq
    var target_len = Int(target_ends[idx])

    var result = semi_global_parasail_gpu[
        DType.int16,
        free_query_start_gaps=True,
        free_query_end_gaps=True,
        free_target_start_gaps=True,
        free_target_end_gaps=True,
    ](
        query_seq_ptr,
        query_len,
        ref_buffer.unsafe_ptr(),
        1024,
        533591,
        idx,
        target_len,
        basic_matrix,
        gap_open_penalty=gap_open_penalty,
        gap_extension_penalty=gap_ext_penalty,
    )

    # Store results
    score_result_buffer[idx] = result.score
    query_end_result_buffer[idx] = Int32(result.query)
    ref_end_result_buffer[idx] = Int32(result.target)


fn process_with_coarse_graining(
    ctx: DeviceContext,
    read dev_query: DeviceBuffer[DType.uint8],
    read dev_ref_buffer: DeviceBuffer[DType.uint8],
    read dev_target_ends: DeviceBuffer[DType.uint32],
    mut dev_score_result_buffer: DeviceBuffer[DType.int32],
    mut dev_query_end_result_buffer: DeviceBuffer[DType.int32],
    mut dev_ref_end_result_buffer: DeviceBuffer[DType.int32],
    read dev_basic_matrix: DeviceBuffer[DType.int8],
    basic_matrix_len: Int,
    query_len: Int,
    target_ends_len: Int,
    threads_to_launch: Int = 15000,
) raises:
    alias BLOCK_SIZE = 32  # Align with warp size
    var num_blocks = ceildiv(threads_to_launch, BLOCK_SIZE)
    print("Launching kernel!")

    # alias BLOCK_SIZE = 1

    var aligner = ctx.compile_function[gpu_align_coarse]()

    ctx.enqueue_function(
        aligner,
        dev_query,
        dev_ref_buffer,
        dev_target_ends,
        dev_score_result_buffer,
        dev_query_end_result_buffer,
        dev_ref_end_result_buffer,
        dev_basic_matrix,
        basic_matrix_len,
        query_len,
        target_ends_len,
        threads_to_launch,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )

    # ctx.synchronize()


# Modified kernel that processes only indices within a specific batch
fn gpu_align_coarse(
    query: DeviceBuffer[DType.uint8],
    ref_buffer: DeviceBuffer[DType.uint8],
    target_ends: DeviceBuffer[DType.uint32],
    score_result_buffer: DeviceBuffer[DType.int32],
    query_end_result_buffer: DeviceBuffer[DType.int32],
    ref_end_result_buffer: DeviceBuffer[DType.int32],
    basic_matrix_values: DeviceBuffer[DType.int8],
    basic_matrix_len: Int,
    query_len: Int,
    target_ends_len: Int,
    thread_count: Int,
):
    # Load scoring matrix into shared memory
    var basic_profile_bytes = stack_allocation[
        576,
        SIMD[DType.int8, 1],
        address_space = AddressSpace.SHARED,
    ]()

    for i in range(0, 576):
        basic_profile_bytes[i] = basic_matrix_values[i]

    var basic_matrix = BasicScoringMatrix[address_space = AddressSpace(3)](
        basic_profile_bytes, 576
    )

    # Load query sequence into shared memory - all threads use the same query
    var query_seq_ptr = stack_allocation[
        200,
        SIMD[DType.uint8, 1],
        address_space = AddressSpace.SHARED,
    ]()

    for i in range(0, query_len):
        query_seq_ptr[i] = query[i]

    barrier()  # Ensure shared memory is fully loaded before proceeding

    # Calculate global thread index
    var thread_id = (block_idx.x * block_dim.x) + thread_idx.x

    # Skip if this thread is outside our desired range
    if thread_id >= thread_count:
        return

    # Hard coded parameters
    var gap_open_penalty = -3
    var gap_ext_penalty = -1

    # Process references in a strided pattern
    # Each thread processes references with indices: thread_id, thread_id + thread_count, thread_id + 2*thread_count, etc.
    for idx in range(thread_id, target_ends_len, thread_count):
        # Get the length of this reference sequence
        var target_len = Int(target_ends[idx])

        # Perform the alignment
        var result = semi_global_parasail_gpu[
            DType.int16,
            free_query_start_gaps=True,
            free_query_end_gaps=True,
            free_target_start_gaps=True,
            free_target_end_gaps=True,
        ](
            query_seq_ptr,
            query_len,
            ref_buffer.unsafe_ptr(),
            1024,
            target_ends_len,
            idx,
            target_len,
            basic_matrix,
            gap_open_penalty=gap_open_penalty,
            gap_extension_penalty=gap_ext_penalty,
        )

        barrier()

        # Store results
        # TODO: move this to after the loop?
        score_result_buffer[idx] = result.score
        query_end_result_buffer[idx] = Int32(result.query)
        ref_end_result_buffer[idx] = Int32(result.target)
        barrier()
