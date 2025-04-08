from ExtraMojo.io.buffered import BufferedReader, BufferedWriter

from ishlib import ByteSpanWriter
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
from ishlib.searcher_settings import SearcherSettings

from algorithm.functional import parallelize
from utils import StringSlice
from sys import stdout, info
from time.time import perf_counter


@value
struct LineAndIndex(SearchableWithIndex):
    var line: List[UInt8]
    var orig_index: UInt

    fn buffer_to_search(ref self) -> Span[UInt8, __origin_of(self)]:
        return rebind[Span[UInt8, __origin_of(self)]](self.line)

    fn original_index(read self) -> UInt:
        return self.orig_index


@value
struct ParallelLineSearchRunner[M: Matcher]:
    var settings: SearcherSettings
    var matcher: M

    fn run_search(mut self) raises:
        # Simple thing first?
        for file in self.settings.files:
            var f = file[]  # force copy
            self.run_search_on_file(f)

    fn run_search_on_file(mut self, file: String) raises:
        # TODO: pass an enocoder to the FastaReader
        var reader = BufferedReader(open(file, "r"))
        var writer = BufferedWriter(stdout)

        var lines = List[LineAndIndex]()
        var bytes_saved = 0
        var buffer = List[UInt8]()
        var line_number = 0

        # TODO: hold onto the non-newline stripped sequence as well for outputting the match color

        var do_work = True
        while do_work:
            buffer.clear()
            if reader.read_until(buffer) == 0:
                do_work = False
            else:
                var line = List[UInt8](capacity=len(buffer))
                for i in range(0, len(buffer)):
                    line.append(
                        self.matcher.convert_ascii_to_encoding(buffer[i])
                    )
                bytes_saved += len(line)
                lines.append(LineAndIndex(line, line_number))

            if bytes_saved >= self.settings.batch_size or not do_work:
                var outputs = List[Optional[ComputedMatchResult]](
                    capacity=len(lines)
                )
                for _ in range(0, len(lines)):
                    outputs.append(None)

                cpu_parallel_starts_ends[M, LineAndIndex](
                    self.matcher,
                    self.settings,
                    lines,
                    outputs,
                )
                # var outputs = self.par(lines)
                for i in range(0, len(outputs)):
                    var m = outputs[i]
                    if not m:
                        continue
                    var r = Pointer.address_of(lines[i])
                    writer.write(
                        file,
                        ":",
                        line_number,
                        " ",
                        # ByteSpanWriter(buffer[:]),
                        ByteSpanWriter(r[].line[0 : m.value().result.start]),
                        "\033[1;31m",
                        ByteSpanWriter(
                            r[].line[
                                m.value().result.start : m.value().result.end
                            ]
                        ),
                        "\033[0m",
                        ByteSpanWriter(r[].line[m.value().result.end :]),
                        "\n",
                    )
                    line_number += 1
                lines.clear()
                bytes_saved = 0
        writer.flush()
        writer.close()


@value
struct GpuParallelLineSearchRunner[
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
        var reader = BufferedReader(open(file, "r"))
        var writer = BufferedWriter(stdout)
        var line_number = 0

        fn write_match[
            mut: Bool, //, o: Origin[mut]
        ](r: Pointer[List[UInt8], o], m: ComputedMatchResult) capturing raises:
            writer.write(
                file,
                ":",
                line_number,
                " ",
                # ByteSpanWriter(buffer[:]),
                ByteSpanWriter(r[][0 : m.result.start]),
                "\033[1;31m",
                ByteSpanWriter(r[][m.result.start : m.result.end]),
                "\033[0m",
                ByteSpanWriter(r[][m.result.end :]),
                "\n",
            )
            line_number += 1

        var cpu_sequences = List[LineAndIndex]()
        var sequences = List[LineAndIndex]()
        var bytes_saved = 0
        var seq_index = 0

        # TODO: hold onto the non-newline stripped sequence as well for outputting the match color

        var start = perf_counter()
        var do_work = True
        var buffer = List[UInt8]()
        while do_work:
            buffer.clear()
            if reader.read_until(buffer) == 0:
                do_work = False
            else:
                var line = List[UInt8](capacity=len(buffer))
                for i in range(0, len(buffer)):
                    line.append(
                        self.matcher.convert_ascii_to_encoding(buffer[i])
                    )
                if len(line) > max_target_length:
                    cpu_sequences.append(LineAndIndex(line, seq_index))
                else:
                    bytes_saved += len(line)
                    sequences.append(LineAndIndex(line, seq_index))

            seq_index += 1
            if bytes_saved >= self.settings.batch_size or not do_work:
                var done_reading = perf_counter()
                print("Time reading", done_reading - start)
                var outputs = gpu_parallel_starts_ends[
                    M,
                    LineAndIndex,
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
                        var r = Pointer(to=sequences[m.value().index].line)
                        write_match(r, m.value())
                    else:
                        var r = Pointer(to=sequences[m.value().index].line)
                        write_match(r, m.value())
                print("write done:", perf_counter() - write_start)
                cpu_sequences.clear()
                sequences.clear()
                bytes_saved = 0
                seq_index = 0
        writer.flush()
        writer.close()
