"""Test program for SSW."""
from sys import stdout, stderr, sizeof
from sys.info import simdwidthof, has_avx512f
from sys.param_env import env_get_int
from time.time import perf_counter
from memory import memcpy
from math import ceildiv

# GPU
from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim, warp, barrier
from gpu.host import DeviceContext, DeviceBuffer
from gpu.memory import AddressSpace, external_memory
from memory import stack_allocation
from layout import Layout, LayoutTensor


from ExtraMojo.cli.parser import OptParser, OptConfig, OptKind, ParsedOpts
from ExtraMojo.io.buffered import BufferedWriter
from ExtraMojo.io.delimited import DelimWriter, ToDelimited
from ishlib.matcher.alignment.ssw_align import (
    ssw_align,
    Profile,
    ScoreSize,
    Alignment,
    ProfileVectors,
    sw,
    ReferenceDirection,
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

# alias SIMD_U8_WIDTH = simdwidthof[UInt8]() // SIMD_MOD
# alias SIMD_U16_WIDTH = simdwidthof[UInt16]() // SIMD_MOD
alias SIMD_U8_WIDTH = 16
alias SIMD_U16_WIDTH = 8


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

    fn __init__(
        out self,
        query_idx: Int,
        target_idx: Int,
        query_len: Int,
        target_len: Int,
        alignment_score: Int,
        query_end: Int,
        target_end: Int,
    ):
        self.query_idx = query_idx
        self.target_idx = target_idx
        self.query_len = query_len
        self.target_len = target_len
        self.alignment_score = alignment_score
        self.query_end = query_end
        self.target_end = target_end

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


# fn gpu_align(
#     query: DeviceBuffer[DType.uint8],
#     ref_buffer: DeviceBuffer[DType.uint8],
#     ref_coords: DeviceBuffer[DType.uint64],
#     fwd_profile: DeviceBuffer[DType.uint16],  # let's just do word for now
#     # rev_profile: DeviceBuffer[DType.uint8],  # lets just do word for now
#     score_result_buffer: DeviceBuffer[DType.uint16],
#     query_end_result_buffer: DeviceBuffer[DType.int32],
#     ref_end_result_buffer: DeviceBuffer[DType.int32],
#     query_len: Int,
#     ref_coords_len: Int,
#     profile_len: Int,
#     profile_bias: UInt16,
# ):
#     alias DT = DType.uint16
#     alias WIDTH = SIMD_U16_WIDTH
#     # Shared memory for the block, holding the profile.
#     # only supports queries of len 1,000 using u16s width 8 (I think)
#     var shared_profile = stack_allocation[
#         600,
#         SIMD[DT, WIDTH],
#         address_space = AddressSpace.SHARED
#         # address_space = AddressSpace.SHARED
#     ]()
#     var profile_span = Span[
#         SIMD[DT, WIDTH],
#         __origin_of(shared_profile),
#         address_space = AddressSpace.SHARED,
#     ](ptr=shared_profile, length=profile_len)
#     var profile_ptr = fwd_profile.unsafe_ptr()
#     var profile_idx = 0
#     for s in range(0, profile_len, WIDTH):
#         var v = profile_ptr.load[width=WIDTH](s)
#         profile_span[profile_idx] = v
#         profile_idx += 1

#     # print(profile_span[0])
#     barrier()

#     var idx = (block_idx.x * block_dim.x) + thread_idx.x

#     if idx > 0 and idx < 3600:
#         pass
#     else:
#         return

#     if idx >= ref_coords_len:
#         return

#     # print(
#     #     "ref cords len:",
#     #     ref_coords_len,
#     #     "idx",
#     #     idx,
#     # )
#     # Hard coded for now:
#     var gap_open_penalty = 3
#     var gap_ext_penalty = 1

#     # Get the start/end coords for this ref seq
#     var coord = LineCoords.from_u64(ref_coords[idx])
#     # print("Loading ref:", coord.start, coord.end)
#     var ref_seq = Span[UInt8, __origin_of(ref_buffer)](
#         ptr=ref_buffer.unsafe_ptr().offset(coord.start),
#         length=Int(coord.end - coord.start),
#     )

#     # Get the query as a span
#     var query_seq = Span[UInt8, __origin_of(query)](
#         ptr=query.unsafe_ptr(), length=query_len
#     )
#     # print(idx, "qlen", query_len, "rlen", len(ref_seq))

#     # Get the profile vectors
#     # TODO: could be reused across a batch that uses the same query?
#     var p_vecs = ProfileVectors[DT, WIDTH](len(query_seq))

#     # print("Processing ref seq:", idx)
#     var result = sw[
#         DT,
#         WIDTH,
#         profile_address_space = AddressSpace(3),
#         allocations_address_space = AddressSpace(5),
#     ](
#         ref_seq,
#         ReferenceDirection.Forward,
#         len(query_seq),
#         gap_open_penalty,
#         gap_ext_penalty,
#         profile_span,
#         p_vecs,
#         -1,
#         profile_bias,
#         15,
#     )

#     print(
#         "index",
#         idx,
#         "Score:",
#         result.best.score,
#         "query_end:",
#         result.best.query,
#         "ref_end:",
#         result.best.reference,
#     )
#     score_result_buffer[idx] = result.best.score - profile_bias
#     query_end_result_buffer[idx] = result.best.query
#     ref_end_result_buffer[idx] = result.best.reference


# Add this function to process references in batches
fn process_in_batches(
    ctx: DeviceContext,
    dev_query: DeviceBuffer[DType.uint8],
    dev_ref_buffer: DeviceBuffer[DType.uint8],
    dev_coords: DeviceBuffer[DType.uint64],
    dev_fwd_profile: DeviceBuffer[DType.uint16],
    dev_score_result_buffer: DeviceBuffer[DType.uint16],
    dev_query_end_result_buffer: DeviceBuffer[DType.int32],
    dev_ref_end_result_buffer: DeviceBuffer[DType.int32],
    query_len: Int,
    ref_coords_len: Int,
    profile_words: Int,
    profile_bias: UInt16,
    batch_size: Int = 3000,
) raises:
    alias BLOCK_SIZE = 32  # Align with warp size

    var num_batches = ceildiv(ref_coords_len, batch_size)
    print(
        "Processing",
        ref_coords_len,
        "references in",
        num_batches,
        "batches of size",
        batch_size,
    )

    for batch in range(num_batches):
        var start_idx = batch * batch_size
        var end_idx = min(start_idx + batch_size, ref_coords_len)
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
        ctx.enqueue_function[gpu_align_batched](
            dev_query,
            dev_ref_buffer,
            dev_coords,
            dev_fwd_profile,
            dev_score_result_buffer,
            dev_query_end_result_buffer,
            dev_ref_end_result_buffer,
            query_len,
            ref_coords_len,
            profile_words,
            profile_bias,
            start_idx,  # Pass the start index
            batch_size_actual,  # Pass the actual batch size
            grid_dim=ceildiv(batch_size_actual, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )

        # Synchronize after each batch to ensure it completes
        ctx.synchronize()


# Modified kernel that processes only indices within a specific batch
fn gpu_align_batched(
    query: DeviceBuffer[DType.uint8],
    ref_buffer: DeviceBuffer[DType.uint8],
    ref_coords: DeviceBuffer[DType.uint64],
    fwd_profile: DeviceBuffer[DType.uint16],
    score_result_buffer: DeviceBuffer[DType.uint16],
    query_end_result_buffer: DeviceBuffer[DType.int32],
    ref_end_result_buffer: DeviceBuffer[DType.int32],
    query_len: Int,
    ref_coords_len: Int,
    profile_len: Int,
    profile_bias: UInt16,
    batch_start_idx: Int,  # Start index for this batch
    batch_size: Int,  # Size of this batch
):
    alias DT = DType.uint16
    alias WIDTH = SIMD_U16_WIDTH
    # Shared memory for the block, holding the profile.
    # only supports queries of len 1,000 using u16s width 8 (I think)
    # var shared_profile = stack_allocation[
    #     600,
    #     SIMD[DT, WIDTH],
    #     address_space = AddressSpace.SHARED
    #     # address_space = AddressSpace.SHARED
    # ]()
    # var profile_span = Span[
    #     SIMD[DT, WIDTH],
    #     __origin_of(shared_profile),
    #     address_space = AddressSpace.SHARED,
    # ](ptr=shared_profile, length=profile_len)
    # var profile_ptr = fwd_profile.unsafe_ptr()
    # var profile_idx = 0
    # for s in range(0, profile_len, WIDTH):
    #     var v = profile_ptr.load[width=WIDTH](s)
    #     profile_span[profile_idx] = v
    #     profile_idx += 1
    # barrier()

    # Calculate global thread index
    var local_idx = (block_idx.x * block_dim.x) + thread_idx.x

    # Convert to global index
    var idx = batch_start_idx + local_idx

    # Skip if this thread is outside the batch range
    if local_idx >= batch_size or idx >= ref_coords_len:
        return

    # # Use fixed-size thread-local memory for profile instead of dynamic List
    var profile_data = stack_allocation[
        576, SIMD[DT, WIDTH]
    ]()  # Adjust size based on your needs
    var profile_span = Span[
        SIMD[DT, WIDTH],
        __origin_of(profile_data),
    ](ptr=profile_data, length=profile_len // WIDTH)

    # Load profile data
    var profile_ptr = fwd_profile.unsafe_ptr()
    for s in range(0, profile_len, WIDTH):
        # if s // WIDTH < len(profile_span):  # Safety check
        var v = profile_ptr.load[width=WIDTH](s)
        profile_span[s // WIDTH] = v

    # Hard coded for now:
    var gap_open_penalty = 3
    var gap_ext_penalty = 1

    # Get the start/end coords for this ref seq
    var coord = LineCoords.from_u64(ref_coords[idx])
    var ref_seq = Span[UInt8, __origin_of(ref_buffer)](
        ptr=ref_buffer.unsafe_ptr().offset(coord.start),
        length=Int(coord.end - coord.start),
    )

    # Get the query as a span
    var query_seq = Span[UInt8, __origin_of(query)](
        ptr=query.unsafe_ptr(), length=query_len
    )

    # Get the profile vectors
    var p_vecs = ProfileVectors[DT, WIDTH](len(query_seq))

    # Call the sw alignment function
    var result = sw[DT, WIDTH](
        ref_seq,
        ReferenceDirection.Forward,
        len(query_seq),
        gap_open_penalty,
        gap_ext_penalty,
        profile_span,  # Use the fixed-size span instead of List
        p_vecs,
        -1,
        profile_bias,
        15,
    )

    # Store results
    score_result_buffer[idx] = result.best.score - profile_bias
    query_end_result_buffer[idx] = result.best.query
    ref_end_result_buffer[idx] = result.best.reference


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

    fn cmp_lens(lhs: FastaRecord, rhs: FastaRecord) capturing -> Bool:
        return len(lhs.seq) > len(rhs.seq)

    sort[cmp_lens](target_seqs)
    var targets = List[ByteFastaRecord](capacity=len(target_seqs))
    var ref_total_bytes = 0
    var ref_coords = List[LineCoords]()
    while len(target_seqs) > 0:
        var t = target_seqs.pop()
        if len(t.seq) > 1024 * 3:
            continue
        # I should be able to pass with ^ here, not sure why I can't
        var r = ByteFastaRecord(t.name, t.seq, matrix)
        var start = ref_total_bytes
        ref_total_bytes += len(r.seq)
        var end = ref_total_bytes
        ref_coords.append(LineCoords(start, end))
        targets.append(r^)
        # print("Remove break here")
        # break
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

    var writer = DelimWriter(
        BufferedWriter(open("./gpu_results.csv", "w")),
        write_header=True,
        delim=",",
    )

    var ctx = DeviceContext()
    var gpu_start = perf_counter()

    var host_ref_buffer = ctx.enqueue_create_host_buffer[DType.uint8](
        ref_total_bytes
    )
    var host_coords = ctx.enqueue_create_host_buffer[DType.uint64](
        len(ref_coords)
    )
    var dev_ref_buffer = ctx.enqueue_create_buffer[DType.uint8](ref_total_bytes)
    var dev_coords = ctx.enqueue_create_buffer[DType.uint64](len(ref_coords))
    ctx.synchronize()
    var host_ref_buffer_ptr = host_ref_buffer.unsafe_ptr()
    var byte_count = 0
    var elem_count = 0
    for target in targets:
        memcpy(
            host_ref_buffer_ptr.offset(byte_count),
            target[].seq.unsafe_ptr(),
            len(target[].seq),
        )
        host_coords[elem_count] = ref_coords[elem_count].to_u64()
        byte_count += len(target[].seq)
        elem_count += 1

    host_ref_buffer.enqueue_copy_to(dev_ref_buffer)
    host_coords.enqueue_copy_to(dev_coords)
    ctx.synchronize()

    # Create the host buffers

    # Create the host buffers
    for i in range(0, len(queries)):
        var query = Pointer.address_of(queries[i])
        var profile = Pointer.address_of(profiles[i])

        var host_query = ctx.enqueue_create_host_buffer[DType.uint8](
            len(query[].seq)
        )
        var host_fwd_profile = ctx.enqueue_create_host_buffer[DType.uint16](
            len(profile[].fwd.profile_word.value()) * SIMD_U16_WIDTH
        )
        var host_fwd_profile_span = Span[UInt16, __origin_of(host_fwd_profile)](
            ptr=host_fwd_profile.unsafe_ptr(),
            length=len(profile[].fwd.profile_word.value()) * SIMD_U16_WIDTH,
        )

        var dev_query = ctx.enqueue_create_buffer[DType.uint8](len(query[].seq))
        var dev_fwd_profile = ctx.enqueue_create_buffer[DType.uint16](
            len(profile[].fwd.profile_word.value()) * SIMD_U16_WIDTH
        )

        # Result buffers
        var host_score_result_buffer = ctx.enqueue_create_host_buffer[
            DType.uint16
        ](len(targets))
        var host_query_end_result_buffer = ctx.enqueue_create_host_buffer[
            DType.int32
        ](len(targets))
        var host_ref_end_result_buffer = ctx.enqueue_create_host_buffer[
            DType.int32
        ](len(targets))

        var dev_score_result_buffer = ctx.enqueue_create_buffer[DType.uint16](
            len(targets)
        )
        var dev_query_end_result_buffer = ctx.enqueue_create_buffer[
            DType.int32
        ](len(targets))
        var dev_ref_end_result_buffer = ctx.enqueue_create_buffer[DType.int32](
            len(targets)
        )

        ctx.enqueue_memset(dev_score_result_buffer, 0)
        ctx.enqueue_memset(dev_query_end_result_buffer, 0)
        ctx.enqueue_memset(dev_ref_end_result_buffer, 0)

        ctx.synchronize()

        memcpy(
            query[].seq.unsafe_ptr(), host_query.unsafe_ptr(), len(query[].seq)
        )

        # Copy profile data
        var profile_words = 0
        for segment in profile[].fwd.profile_word.value():
            for i in range(0, len(segment[])):
                host_fwd_profile_span[profile_words] = segment[][i]
                profile_words += 1

        host_query.enqueue_copy_to(dev_query)
        host_fwd_profile.enqueue_copy_to(dev_fwd_profile)

        # Use batched processing instead of single kernel launch
        process_in_batches(
            ctx,
            dev_query,
            dev_ref_buffer,
            dev_coords,
            dev_fwd_profile,
            dev_score_result_buffer,
            dev_query_end_result_buffer,
            dev_ref_end_result_buffer,
            len(query[].seq),
            len(ref_coords),
            profile_words,
            profile[].fwd.bias.cast[DType.uint16](),
            batch_size=3000,
        )

        # Copy results back to host
        dev_score_result_buffer.enqueue_copy_to(host_score_result_buffer)
        dev_query_end_result_buffer.enqueue_copy_to(
            host_query_end_result_buffer
        )
        dev_ref_end_result_buffer.enqueue_copy_to(host_ref_end_result_buffer)

        # Write results
        for j in range(0, len(ref_coords)):
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
    print("GPU TOOK:", gpu_end - gpu_start)
    writer.flush()
