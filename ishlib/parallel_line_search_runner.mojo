from ExtraMojo.io.buffered import BufferedWriter

from ishlib import ByteSpanWriter, RecordType
from ishlib.peek_file import peek_file, PeekFindings
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
from ishlib.vendor.log import Logger
from ishlib.vendor.kseq import (
    BufferedReader,
    ByteString,
    FastxReader,
    SearchChar,
)
from ishlib.vendor.zlib import GZFile


from algorithm.functional import parallelize
from pathlib import Path
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
            var peek = peek_file[record_type = RecordType.LINE](f)
            Logger.debug("Suggested length:", peek.suggested_max_length)
            if peek.is_binary:
                Logger.warn("Skipping binary file:", file[])
                continue
            self.run_search_on_file(f)

    fn run_search_on_file(mut self, path: Path) raises:
        var file = String(path)
        # TODO: pass an enocoder to the FastaReader
        var reader = BufferedReader(GZFile(file, "rb"))
        var writer = BufferedWriter(stdout)

        var lines = List[LineAndIndex]()
        var bytes_saved = 0
        # var buffer = List[UInt8]()
        var buffer = ByteString()
        var batch_line_number = 0
        var global_lines_seen = 0
        # TODO: hold onto the non-newline stripped sequence as well for outputting the match color

        var do_work = True
        while do_work:
            buffer.clear()
            if reader.read_until[delim = SearchChar.Newline](buffer) <= 0:
                do_work = False
            else:
                var line = List[UInt8](capacity=len(buffer))
                for i in range(0, len(buffer)):
                    line.append(
                        self.matcher.convert_ascii_to_encoding(buffer[i])
                    )
                bytes_saved += len(line)
                lines.append(LineAndIndex(line, batch_line_number))
                batch_line_number += 1

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

                for i in range(0, len(outputs)):
                    var m = outputs[i]
                    if not m:
                        continue
                    var r = Pointer.address_of(lines[i])

                    for i in range(0, len(r[].line)):
                        r[].line[i] = self.matcher.convert_encoding_to_ascii(
                            r[].line[i]
                        )
                    if self.settings.tty_info.is_a_tty:
                        writer.write(
                            file,
                            ":",
                            r[].orig_index + global_lines_seen + 1,
                            " ",
                            ByteSpanWriter(
                                r[].line[0 : m.value().result.start]
                            ),
                            "\033[1;31m",
                            ByteSpanWriter(
                                r[].line[
                                    m.value()
                                    .result.start : m.value()
                                    .result.end
                                ]
                            ),
                            "\033[0m",
                            ByteSpanWriter(r[].line[m.value().result.end :]),
                            "\n",
                        )
                    else:
                        writer.write(
                            ByteSpanWriter(r[].line[:]),
                            "\n",
                        )
                lines.clear()
                bytes_saved = 0
                global_lines_seen += batch_line_number
                batch_line_number = 0
        writer.flush()
        writer.close()


@value
struct GpuParallelLineSearchRunner[
    M: GpuMatcher,
    max_query_length: UInt = 200,
    max_target_length: UInt = 1024,
]:
    """Parallel runner that _can_ make use of GPU as well as CPU."""

    var settings: SearcherSettings
    var matcher: M

    fn __init__(out self, settings: SearcherSettings, matcher: M) raises:
        self.settings = settings
        self.matcher = matcher

    fn run_search(mut self) raises:
        # Peek the first file to get the suggested size, then use that for all of them.
        # Still peek each for binary

        # First non-binary file
        var files = self.settings.files
        var first_peek = peek_file[
            record_type = RecordType.LINE, check_record_size=True
        ](files[0])
        if first_peek.is_binary:
            for i in range(1, len(files)):
                first_peek = peek_file[
                    record_type = RecordType.LINE, check_record_size=True
                ](files[i])

        Logger.debug("Suggested length of:", first_peek.suggested_max_length)
        # Create ctxs
        if first_peek.suggested_max_length <= 128:
            var ctxs = self.create_ctxs[max_query_length, 128]()
            self.search_files[
                max_query_length=max_query_length, max_target_length=128
            ](files, ctxs)
        elif first_peek.suggested_max_length <= 256:
            var ctxs = self.create_ctxs[max_query_length, 256]()
            self.search_files[
                max_query_length=max_query_length, max_target_length=256
            ](files, ctxs)
        elif first_peek.suggested_max_length <= 512:
            var ctxs = self.create_ctxs[max_query_length, 512]()
            self.search_files[
                max_query_length=max_query_length, max_target_length=512
            ](files, ctxs)
        elif first_peek.suggested_max_length <= 1024:
            var ctxs = self.create_ctxs[max_query_length, 1024]()
            self.search_files[
                max_query_length=max_query_length, max_target_length=1024
            ](files, ctxs)
        elif first_peek.suggested_max_length <= 2048:
            var ctxs = self.create_ctxs[max_query_length, 2048]()
            self.search_files[
                max_query_length=max_query_length, max_target_length=2048
            ](files, ctxs)
        elif first_peek.suggested_max_length <= 4096:
            var ctxs = self.create_ctxs[max_query_length, 4096]()
            self.search_files[
                max_query_length=max_query_length, max_target_length=4096
            ](files, ctxs)
        else:
            # TODO: do we actually have an upper limit?
            Logger.warn(
                "Longer line lengths that nicely supported, more work will"
                " be sent to CPU, concider running with max-gpus set to 0."
            )
            var ctxs = self.create_ctxs[max_query_length, 4096]()
            self.search_files[
                max_query_length=max_query_length, max_target_length=4096
            ](files, ctxs)

    fn create_ctxs[
        max_query_length: UInt = 200, max_target_length: UInt = 1024
    ](mut self) raises -> List[
        SearcherDevice[
            M.batch_match_coarse[max_query_length, max_target_length]
        ]
    ]:
        var ctxs = SearcherDevice[
            M.batch_match_coarse[max_query_length, max_target_length]
        ].create_devices(
            self.settings.batch_size,
            len(self.settings.pattern),
            self.matcher.matrix_len(),
            max_target_length=max_target_length,
            max_devices=self.settings.max_gpus,
        )
        return ctxs

    fn search_files[
        max_query_length: UInt = 200, max_target_length: UInt = 1024
    ](
        mut self,
        paths: List[Path],
        mut ctxs: List[
            SearcherDevice[
                M.batch_match_coarse[max_query_length, max_target_length]
            ]
        ],
    ) raises:
        for i in range(0, len(paths)):
            var f = paths[i]  # force copy
            Logger.debug("Processing", f)
            if i > 0:
                # Can skip the first peek since we've already checked it.
                var peek = peek_file[
                    record_type = RecordType.LINE, check_record_size=False
                ](f)
                if peek.is_binary:
                    Logger.warn("Skipping binary file:", paths[i])
                    continue
            self.run_search_on_file[max_query_length, max_target_length](
                f, ctxs
            )

    fn run_search_on_file[
        max_query_length: UInt = 200, max_target_length: UInt = 1024
    ](
        mut self,
        path: Path,
        mut ctxs: List[
            SearcherDevice[
                M.batch_match_coarse[max_query_length, max_target_length]
            ]
        ],
    ) raises:
        var file = String(path)
        var reader = BufferedReader(GZFile(file, "rb"))
        var writer = BufferedWriter(stdout)

        fn write_match[
            mut: Bool, //, o: Origin[mut]
        ](
            r: Pointer[List[UInt8], o], m: ComputedMatchResult, orig_index: Int
        ) capturing raises:
            if self.settings.tty_info.is_a_tty:
                writer.write(
                    file,
                    ":",
                    orig_index + 1,
                    " ",
                    # ByteSpanWriter(buffer[:]),
                    ByteSpanWriter(r[][0 : m.result.start]),
                    "\033[1;31m",
                    ByteSpanWriter(r[][m.result.start : m.result.end]),
                    "\033[0m",
                    ByteSpanWriter(r[][m.result.end :]),
                    "\n",
                )
            else:
                writer.write(
                    ByteSpanWriter(r[][:]),
                    "\n",
                )

        var cpu_sequences = List[LineAndIndex]()
        var sequences = List[LineAndIndex]()
        var bytes_saved = 0
        var seq_index = 0
        var global_seqs_seen = 0

        # TODO: hold onto the non-newline stripped sequence as well for outputting the match color

        var start = perf_counter()
        var do_work = True
        # var buffer = List[UInt8]()
        var buffer = ByteString()
        while do_work:
            buffer.clear()
            if reader.read_until[delim = SearchChar.Newline](buffer) <= 0:
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
                    bytes_saved += max_target_length
                    sequences.append(LineAndIndex(line, seq_index))

            seq_index += 1
            if (
                bytes_saved >= (self.settings.batch_size - max_target_length)
                or not do_work
            ):
                var done_reading = perf_counter()
                Logger.debug("Time reading", done_reading - start)
                var outputs = gpu_parallel_starts_ends[
                    M,
                    LineAndIndex,
                    max_query_length,
                    max_target_length,
                ](
                    ctxs,
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
                        for i in range(0, len(r[])):
                            r[][i] = self.matcher.convert_encoding_to_ascii(
                                r[][i]
                            )
                        write_match(
                            r,
                            m.value(),
                            sequences[m.value().index].orig_index
                            + global_seqs_seen,
                        )
                    else:
                        var r = Pointer(to=cpu_sequences[m.value().index].line)
                        for i in range(0, len(r[])):
                            r[][i] = self.matcher.convert_encoding_to_ascii(
                                r[][i]
                            )
                        write_match(
                            r,
                            m.value(),
                            cpu_sequences[m.value().index].orig_index
                            + global_seqs_seen,
                        )
                Logger.debug("write done:", perf_counter() - write_start)
                cpu_sequences.clear()
                sequences.clear()
                bytes_saved = 0
                global_seqs_seen += seq_index
                seq_index = 0
        writer.flush()
        writer.close()
