"""Test program for SSW."""
from algorithm.functional import parallelize
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from math import ceildiv
from sys import stdout, stderr
from sys.info import simdwidthof, has_avx512f, alignof, num_physical_cores
from sys.param_env import env_get_int
from time.time import perf_counter

from gpu.host import DeviceContext


from ExtraMojo.cli.parser import OptParser, OptConfig, OptKind, ParsedOpts
from ExtraMojo.io.buffered import BufferedWriter
from ExtraMojo.io.delimited import DelimWriter, ToDelimited

from ishlib.formats.fasta import FastaRecord
from ishlib.gpu.searcher_device import SearcherDevice
from ishlib.matcher import Searchable
from ishlib.matcher.alignment.local_aln.striped import (
    ScoreSize,
    Alignment,
)
from ishlib.matcher.alignment.scoring_matrix import (
    ScoringMatrix,
    BasicScoringMatrix,
    MatrixKind,
    BLOSUM62,
    NUM_TO_AA,
)
from ishlib.matcher.striped_semi_global_matcher import StripedSemiGlobalMatcher
from ishlib.searcher_settings import SemiGlobalEndsFreeness
from ishlib.matcher.alignment.striped_utils import AlignmentResult
from ishlib.matcher.alignment.semi_global_aln.basic import (
    SGResult,
)
from ishlib.vendor.log import Logger


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
            "devices",
            OptKind.IntLike,
            default_value=String("1"),
            description="Num GPUs to use.",
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

    return parser.parse_sys_args()


@value
struct ByteFastaRecord(Searchable):
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

    fn buffer_to_search(ref self) -> Span[UInt8, __origin_of(self)]:
        return rebind[Span[UInt8, __origin_of(self)]](self.seq)


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

        return Self(
            total_query_seqs // len(results),
            total_target_seqs // len(results),
            query_len,
            matrix,
            gap_open,
            gap_extend,
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
    var devices = opts.get_int("devices")

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

    bench_basic_semi_global_gpu_parallel(
        target_file,
        query_file,
        metrics_file,
        output_file,
        matrix,
        iterations,
        matrix_name,
        gap_open_score,
        gap_extension_score,
        devices,
    )


fn bench_basic_semi_global_gpu_parallel[
    write_output_metrics: Bool = False
](
    target_file: String,
    query_file: String,
    metrics_file: String,
    output_file: String,
    matrix: ScoringMatrix,
    iterations: Int,
    matrix_name: String,
    gap_open_score: Int,
    gap_extension_score: Int,
    devices: Int,
) raises:
    alias MAX_GPU_LENGTH = 1024

    if devices > 1:
        Logger.error("Only supports 1 device for using Bench")
        raise "Only supports 1 device for using Bench"
    ################
    # Setup
    ################
    # Read the fastas and encode the sequences
    var prep_start = perf_counter()
    var target_seqs = FastaRecord.slurp_fasta(target_file)

    var matrix_kind = MatrixKind.from_str(matrix_name)

    var query_seqs = FastaRecord.slurp_fasta(query_file)
    var queries = List[ByteFastaRecord](capacity=len(query_seqs))
    while len(query_seqs) > 0:
        var q = query_seqs.pop()
        queries.append(ByteFastaRecord(q.name, q.seq, matrix))
    queries.reverse()

    var qlen = len(queries[0].seq)

    var work = 0  # total cells processed
    var targets = List[ByteFastaRecord](capacity=len(target_seqs))
    var target_ends = List[Int]()
    var target_bytes = 0
    while len(target_seqs) > 0:
        var t = target_seqs.pop()
        if len(t.seq) > MAX_GPU_LENGTH:
            continue
        var r = ByteFastaRecord(t.name, t.seq, matrix)
        target_ends.append(len(t.seq))
        target_bytes += len(t.seq)
        targets.append(r^)
        work += qlen * len(t.seq)

    # Create query profiles
    var prep_end = perf_counter()
    Logger.timing("Setup Time:", prep_end - prep_start)

    ################
    # Copy to GPU
    ################

    var num_devices = devices
    var targets_per_device = ceildiv(len(target_ends), num_devices)
    Logger.timing("Devices:", num_devices)
    Logger.timing("Total targets:", len(target_ends))
    Logger.timing("Targets per device:", targets_per_device)

    var ctxs = SearcherDevice[
        StripedSemiGlobalMatcher.batch_match_coarse[200, MAX_GPU_LENGTH]
    ].create_devices(
        targets_per_device * MAX_GPU_LENGTH,
        len(queries[0].seq),
        len(matrix),
        max_target_length=MAX_GPU_LENGTH,
        max_devices=devices,
    )

    # Create Buffers
    var host_buffer_create_start = perf_counter()
    for ctx_id in range(0, num_devices):
        ctxs[ctx_id].set_block_info(
            targets_per_device,
            len(queries[0].seq),
            len(matrix),
            matrix_kind,
            gap_open_score,
            gap_extension_score,
            SemiGlobalEndsFreeness.TTFF,
            max_target_length=MAX_GPU_LENGTH,
        )
        ctxs[ctx_id].host_create_input_buffers()

    for ctx in ctxs:
        ctx[].synchronize()
    var host_buffers_created = perf_counter()
    Logger.timing(
        "Host buffer creation time:",
        host_buffers_created - host_buffer_create_start,
    )

    # Fill in input data
    var buffer_fill_start = perf_counter()
    var amounts = List[Tuple[Int, Int]]()
    for i in range(0, len(target_ends), targets_per_device):
        amounts.append((i, min(i + targets_per_device, len(targets))))

    fn bcmp_lens(lhs: ByteFastaRecord, rhs: ByteFastaRecord) capturing -> Bool:
        return len(lhs.seq) < len(rhs.seq)

    @parameter
    fn copy_data(i: Int):
        start, end = amounts[i]
        var local_targets = targets[start:end]
        sort[bcmp_lens](local_targets)

        ctxs[i].set_host_inputs(
            Span(queries[0].seq),
            matrix.values.unsafe_ptr(),
            len(matrix),
            local_targets,
        )

    parallelize[copy_data](num_devices)
    var buffers_filled = perf_counter()
    Logger.timing("Buffer fill time:", buffers_filled - buffer_fill_start)

    # Don't count the data setup time, but do count the transfer time
    var gpu_start = perf_counter()

    # Launch Kernel
    for ctx in ctxs:
        ctx[].device_create_input_buffers()
        ctx[].copy_inputs_to_device()
        ctx[].device_create_output_buffers()

    for ctx in ctxs:
        ctx[].synchronize()

    var b = Bench()
    var gcups_metric = BenchMetric(5, "Giga-Cell Updates per second", "GCUPS")
    var gcups = ThroughputMeasure(gcups_metric, work)
    var bytes_ = ThroughputMeasure(BenchMetric.bytes, target_bytes)

    @parameter
    @always_inline
    fn bench_gpu[
        block_size: Int, threads_to_launch: Int
    ](mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(gpu_ctx: DeviceContext) raises:
            ctxs[0].launch_kernel[block_size=block_size](threads_to_launch)

        # Check at start of fn that we only have 1 devices specified
        b.iter_custom[kernel_launch](ctxs[0].ctx)

    """
    To further change how things are run, modify the `ctxs[].launch_kernel` on the `SearcherDevice`.

    Current default is for 15000 threads, block_size 32, tuned on an L4.
    
    The primary difference between this example, and how it's run in `ish` is that in `ish`
    we process batches of records at a time, so the SearcherDevice manages and reuses buffers
    to avoid allocating for each batch since they are mostly the same size.
    """
    b.bench_function[bench_gpu[32, 10000]](
        BenchId("coarse graining, 32x10000", "gpu"),
        bytes_,
        gcups,
    )
    b.bench_function[bench_gpu[32, 15000]](
        BenchId("coarse graining, 32x15000", "gpu"),
        bytes_,
        gcups,
    )
    b.bench_function[bench_gpu[32, 30000]](
        BenchId("coarse graining, 32x30000", "gpu"),
        bytes_,
        gcups,
    )
    b.config.verbose_metric_names = False
    print(b)

    # TODO: is this even needed?
    for ctx in ctxs:
        ctx[].synchronize()

    for ctx in ctxs:
        ctx[].host_create_output_buffers()
        ctx[].copy_outputs_to_host()

    for ctx_id in range(0, num_devices):
        ctxs[ctx_id].synchronize()

    var kernel_done = perf_counter()
    Logger.timing(
        "GPU processing time (with cpu):", kernel_done - buffers_filled
    )

    @parameter
    if write_output_metrics:
        write_outputs[200, MAX_GPU_LENGTH](
            ctxs,
            targets,
            queries,
            gpu_start,
            target_file,
            query_file,
            metrics_file,
            output_file,
            matrix,
            iterations,
            matrix_name,
            gap_open_score,
            gap_extension_score,
            devices,
        )


fn write_outputs[
    query_len: Int, target_len: Int
](
    ctxs: List[
        SearcherDevice[
            StripedSemiGlobalMatcher.batch_match_coarse[query_len, target_len]
        ]
    ],
    read targets: List[ByteFastaRecord],
    read queries: List[ByteFastaRecord],
    gpu_start: Float64,
    target_file: String,
    query_file: String,
    metrics_file: String,
    output_file: String,
    matrix: ScoringMatrix,
    iterations: Int,
    matrix_name: String,
    gap_open_score: Int,
    gap_extension_score: Int,
    devices: Int,
) raises:
    var writer = DelimWriter(
        BufferedWriter(open(output_file, "w")),
        write_header=True,
        delim=",",
    )

    # Write output
    var write_start = perf_counter()
    var work = 0
    var items = 0
    var total_items = 0
    for ctx in ctxs:
        var end = min(
            total_items + ctx[].block_info.value().num_targets, len(targets)
        )

        for j in range(0, end):
            var query_len = len(queries[0].seq)
            var target_len = ctx[].host_target_ends.value().as_span()[j]
            score = ctx[].host_scores.value().as_span()[j]
            query_end = ctx[].host_query_ends.value().as_span()[j]
            target_end = ctx[].host_target_ends.value().as_span()[j]
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

    # Result writing time included for parity with other methods
    var write_end = perf_counter()
    Logger.timing("Write time:", write_end - write_start)
    writer.flush()
    var elapsed = write_end - gpu_start
    var cells_per_second = work / elapsed
    Logger.timing("Num Devices:", devices)
    Logger.timing("GPU TOOK:", elapsed)
    Logger.timing("Algo: basic-semi-global-gpu")
    Logger.timing("Ran in", elapsed, "seconds")
    Logger.timing("Total cells updated :", work)
    Logger.timing("Cells per second:", cells_per_second)
    Logger.timing("GCUPs:", cells_per_second / 1000000000)
    Logger.timing(
        "gpu GCUPs:",
        (work / (write_end - gpu_start) / 1000000000),
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
