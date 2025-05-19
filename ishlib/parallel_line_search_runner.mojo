from ExtraMojo.io import MovableWriter
from ExtraMojo.io.buffered import BufferedWriter

from ishlib import RED, PURPLE, GREEN
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

    fn run_search[
        W: MovableWriter
    ](mut self, mut writer: BufferedWriter[W]) raises:
        # Simple thing first?
        for file in self.settings.files:
            var f = file[]  # force copy
            var peek = peek_file[record_type = RecordType.LINE](f)
            Logger.debug("Suggested length:", peek.suggested_max_length)
            if peek.is_binary and self.settings.verbose:
                Logger.warn("Skipping binary file:", file[])
                continue
            self.run_search_on_file(f, writer)

    fn run_search_on_file[
        W: MovableWriter
    ](mut self, path: Path, mut writer: BufferedWriter[W]) raises:
        var file = String(path)
        var reader = BufferedReader(GZFile(file, "rb"))

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
            if reader.read_until[delim = SearchChar.Newline](buffer) < 0:
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
                    var r = Pointer(to=lines[i])

                    for i in range(0, len(r[].line)):
                        r[].line[i] = self.matcher.convert_encoding_to_ascii(
                            r[].line[i]
                        )
                    if (
                        self.settings.tty_info.is_a_tty
                        and self.settings.is_output_stdout()
                    ):
                        writer.write(
                            PURPLE,
                            file,
                            RESET,
                            ":",
                            GREEN,
                            r[].orig_index + global_lines_seen + 1,
                            RESET,
                            ": ",
                            ByteSpanWriter(
                                r[].line[0 : m.value().result.start]
                            ),
                            RED,
                            ByteSpanWriter(
                                r[].line[
                                    m.value()
                                    .result.start : m.value()
                                    .result.end
                                ]
                            ),
                            RESET,
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

    fn run_search[
        W: MovableWriter
    ](mut self, mut writer: BufferedWriter[W]) raises:
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
        @parameter
        @always_inline
        fn choose_max_target_length(suggested_max_length: Int) raises:
            alias MAX_TARGET_LENGTHS = List(128, 256, 512, 1024, 2048, 4096)

            @parameter
            for i in range(0, len(MAX_TARGET_LENGTHS)):
                alias max_target_length = MAX_TARGET_LENGTHS[i]
                if suggested_max_length <= max_target_length:
                    var ctxs = self.create_ctxs[
                        max_query_length, max_target_length
                    ]()
                    self.search_files[
                        W,
                        max_query_length=max_query_length,
                        max_target_length=max_target_length,
                    ](files, ctxs, writer)
                    return

            Logger.warn(
                "Longer line lengths than supported, more work will"
                " be sent to CPU, consider running with max-gpus set to 0."
            )
            var ctxs = self.create_ctxs[max_query_length, 4096]()
            self.search_files[
                W, max_query_length=max_query_length, max_target_length=4096
            ](files, ctxs, writer)

        choose_max_target_length(first_peek.suggested_max_length)

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
        W: MovableWriter,
        max_query_length: UInt = 200,
        max_target_length: UInt = 1024,
    ](
        mut self,
        paths: List[Path],
        mut ctxs: List[
            SearcherDevice[
                M.batch_match_coarse[max_query_length, max_target_length]
            ]
        ],
        mut writer: BufferedWriter[W],
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
            self.run_search_on_file[W, max_query_length, max_target_length](
                f, ctxs, writer
            )

    fn run_search_on_file[
        W: MovableWriter,
        max_query_length: UInt = 200,
        max_target_length: UInt = 1024,
    ](
        mut self,
        path: Path,
        mut ctxs: List[
            SearcherDevice[
                M.batch_match_coarse[max_query_length, max_target_length]
            ]
        ],
        mut writer: BufferedWriter[W],
    ) raises:
        var file = String(path)
        var reader = BufferedReader(GZFile(file, "rb"))

        fn write_match[
            mut: Bool, //, o: Origin[mut]
        ](
            r: Pointer[List[UInt8], o], m: ComputedMatchResult, orig_index: Int
        ) capturing raises:
            if (
                self.settings.tty_info.is_a_tty
                and self.settings.is_output_stdout()
            ):
                writer.write(
                    PURPLE,
                    file,
                    RESET,
                    ":",
                    GREEN,
                    orig_index + 1,
                    RESET,
                    ": ",
                    # ByteSpanWriter(buffer[:]),
                    ByteSpanWriter(r[][0 : m.result.start]),
                    RED,
                    ByteSpanWriter(r[][m.result.start : m.result.end]),
                    RESET,
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
            if reader.read_until[delim = SearchChar.Newline](buffer) < 0:
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
                bytes_saved
                >= (self.settings.readable_batch_size() - max_target_length)
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
