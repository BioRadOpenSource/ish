from ExtraMojo.io.buffered import BufferedWriter

from time.time import perf_counter

from ishlib.formats.fasta import (
    FastaReader,
    BorrowedFastaRecord,
    ByteFastaRecord,
)
from ishlib.searcher_settings import SearcherSettings
from ishlib.cpu.process_batch import (
    parallel_starts_ends as cpu_parallel_starts_ends,
)
from ishlib.gpu import has_gpu
from ishlib.gpu.process_batch import (
    parallel_starts_ends as gpu_parallel_starts_ends,
)
from ishlib.gpu.searcher_device import SearcherDevice
from ishlib.matcher import (
    Matcher,
    MatchResult,
    SearchableWithIndex,
    GpuMatcher,
    ComputedMatchResult,
    WhereComputed,
)
from ishlib import ByteSpanWriter
from ishlib.matcher.alignment.striped_utils import AlignmentResult
from ishlib.vendor.kseq import BufferedReader, FastxReader
from ishlib.vendor.zlib import GZFile

from algorithm.functional import parallelize
from math import ceildiv
from utils import StringSlice
from sys import stdout, info


@value
struct SeqAndIndex(SearchableWithIndex):
    var seq: ByteFastaRecord
    var orig_index: UInt

    fn buffer_to_search(ref self) -> Span[UInt8, __origin_of(self)]:
        return rebind[Span[UInt8, __origin_of(self)]](
            self.seq.buffer_to_search()
        )

    fn original_index(read self) -> UInt:
        return self.orig_index


@value
struct ParallelFastaSearchRunner[M: Matcher]:
    var settings: SearcherSettings
    var matcher: M

    fn run_search(mut self) raises:
        # Simple thing first?
        for file in self.settings.files:
            var f = file[]  # force copy
            self.run_search_on_file(f)

    fn run_search_on_file(mut self, file: String) raises:
        # TODO: pass an enocoder to the FastaReader
        var reader = FastxReader[read_comment=False](
            BufferedReader(GZFile(file, "r"))
        )
        var writer = BufferedWriter(stdout)

        var sequences = List[SeqAndIndex]()
        var bytes_saved = 0
        var seq_number = 0

        # TODO: hold onto the non-newline stripped sequence as well for outputting the match color

        var do_work = True
        while do_work:
            var ret = reader.read()

            if ret <= 0:
                do_work = False
            else:
                var record = ByteFastaRecord(
                    List(reader.name.as_span()), List(reader.seq.as_span())
                )
                bytes_saved += record.size_in_bytes()
                sequences.append(SeqAndIndex(record, seq_number))
                seq_number += 1

            if bytes_saved >= self.settings.batch_size or not do_work:
                var outputs = List[Optional[ComputedMatchResult]](
                    capacity=len(sequences)
                )
                for _ in range(0, len(sequences)):
                    outputs.append(None)

                cpu_parallel_starts_ends[M, SeqAndIndex](
                    self.matcher,
                    self.settings,
                    sequences,
                    outputs,
                )
                for i in range(0, len(outputs)):
                    var m = outputs[i]
                    if not m:
                        continue
                    var r = Pointer.address_of(sequences[i])
                    writer.write(">")
                    writer.write_bytes(r[].seq.name)
                    writer.write("\n")
                    writer.write_bytes(r[].seq.seq[0 : m.value().result.start])
                    writer.write("\033[1;31m")
                    writer.write_bytes(
                        r[].seq.seq[
                            m.value().result.start : m.value().result.end
                        ]
                    )
                    writer.write()
                    writer.write("\033[0m")
                    writer.write_bytes(r[].seq.seq[m.value().result.end :])
                    writer.write("\n")
                sequences.clear()
                bytes_saved = 0
                seq_number = 0
        writer.flush()
        writer.close()


@value
struct GpuParallelFastaSearchRunner[
    M: GpuMatcher,
    max_matrix_length: UInt = 576,
    max_query_length: UInt = 200,
    max_target_length: UInt = 1024,
]:
    """Parallel runner that _can_ make use of GPU as well as CPU."""

    var settings: SearcherSettings
    var matcher: M
    var ctxs: List[
        SearcherDevice[
            M.batch_match_coarse[
                max_matrix_length, max_query_length, max_target_length
            ]
        ]
    ]

    fn __init__(out self, settings: SearcherSettings, matcher: M) raises:
        self.settings = settings
        self.matcher = matcher
        self.ctxs = SearcherDevice[
            M.batch_match_coarse[
                max_matrix_length, max_query_length, max_target_length
            ]
        ].create_devices()

    fn run_search(mut self) raises:
        # Simple thing first?
        for file in self.settings.files:
            var f = file[]  # force copy
            self.run_search_on_file(f)

    fn run_search_on_file(mut self, file: String) raises:
        # TODO: Split out the too-long seqs
        # TODO: pass an enocoder to the FastaReader
        var reader = FastxReader[read_comment=False](
            BufferedReader(GZFile(file, "r"))
        )
        var writer = BufferedWriter(stdout)

        fn write_match[
            mut: Bool, //, o: Origin[mut]
        ](
            r: Pointer[ByteFastaRecord, o], m: ComputedMatchResult
        ) capturing raises:
            writer.write(">")
            writer.write_bytes(r[].name)
            writer.write("\n")
            writer.write_bytes(r[].seq[0 : m.result.start])
            # writer.write("\033[1;31m")
            writer.write_bytes(r[].seq[m.result.start : m.result.end])
            writer.write()
            # writer.write("\033[0m")
            writer.write_bytes(r[].seq[m.result.end :])
            writer.write("\n")

        var cpu_sequences = List[SeqAndIndex]()
        var sequences = List[SeqAndIndex]()
        var bytes_saved = 0
        var seq_index = 0

        # TODO: hold onto the non-newline stripped sequence as well for outputting the match color

        var start = perf_counter()
        var do_work = True
        while do_work:
            var ret = reader.read()

            if ret <= 0:
                do_work = False
            else:
                if len(reader.seq) > max_target_length:
                    cpu_sequences.append(
                        SeqAndIndex(
                            ByteFastaRecord(
                                List(reader.name.as_span()),
                                List(reader.seq.as_span()),
                            ),
                            seq_index,
                        )
                    )
                else:
                    var record = ByteFastaRecord(
                        List(reader.name.as_span()),
                        List(reader.seq.as_span()),
                    )
                    bytes_saved += record.size_in_bytes()
                    sequences.append(SeqAndIndex(record, seq_index))

            seq_index += 1
            if bytes_saved >= self.settings.batch_size or not do_work:
                var done_reading = perf_counter()
                print("Time reading", done_reading - start)
                var outputs = gpu_parallel_starts_ends[
                    M,
                    SeqAndIndex,
                    max_matrix_length,
                    max_query_length,
                    max_target_length,
                ](
                    self.ctxs,
                    self.matcher,
                    self.settings,
                    sequences,
                    cpu_sequences,
                )
                var write_start = perf_counter()
                for i in range(0, len(outputs)):
                    var m = outputs[i]
                    if not m:
                        continue

                    if m.value().where_computed == WhereComputed.Gpu:
                        var r = Pointer(to=sequences[m.value().index].seq)
                        write_match(r, m.value())
                    else:
                        var r = Pointer(to=sequences[m.value().index].seq)
                        write_match(r, m.value())
                print("write done:", perf_counter() - write_start)
                cpu_sequences.clear()
                sequences.clear()
                bytes_saved = 0
                seq_index = 0
        writer.flush()
        writer.close()
