from ExtraMojo.io import MovableWriter
from ExtraMojo.io.buffered import BufferedWriter

from time.time import perf_counter

from ishlib import RED, PURPLE, GREEN
from ishlib.formats.fasta import (
    FastaReader,
    BorrowedFastaRecord,
    ByteFastaRecord,
)
from ishlib.searcher_settings import (
    SearcherSettings,
)
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
from ishlib import ByteSpanWriter, RecordType
from ishlib.peek_file import peek_file
from ishlib.matcher.alignment.striped_utils import AlignmentResult
from ishlib.vendor.kseq import BufferedReader, FastxReader
from ishlib.vendor.zlib import GZFile
from ishlib.vendor.log import Logger
from ishlib.vendor.pingpong import run

from algorithm.functional import parallelize
from math import ceildiv
from pathlib import Path
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

    fn run_search[
        W: MovableWriter
    ](mut self, mut writer: BufferedWriter[W]) raises:
        # Simple thing first?
        for file in self.settings.files:
            var f = file[]  # force copy
            var peek = peek_file[record_type = RecordType.FASTA](f)
            Logger.debug("Suggested length:", peek.suggested_max_length)
            if peek.is_binary and self.settings.verbose:
                Logger.warn("Skipping binary file:", file[])
            Logger.debug("Processing", f)
            self.run_search_on_file_ping_pong(f, writer)

    fn run_search_on_file_ping_pong[
        W: MovableWriter
    ](mut self, file: Path, mut writer: BufferedWriter[W]) raises:
        var reader = FastxReader[read_comment=False](
            BufferedReader(GZFile(String(file), "r"))
        )

        var a_sequences = List[SeqAndIndex]()
        var a_bytes_saved = 0
        var a_seq_number = 0
        var a_do_work = True

        var b_sequences = List[SeqAndIndex]()
        var b_bytes_saved = 0
        var b_seq_number = 0
        var b_do_work = True

        @parameter
        fn a_read() capturing -> Int:
            print("a read")
            while a_do_work:
                var ret: Int = -1
                try:
                    ret = reader.read()
                except:
                    Logger.error("Failed to read from file", String(file))
                    return -1

                if ret == 0:
                    a_do_work = False
                elif ret < 0:
                    return -1
                else:
                    var seq = List[UInt8](capacity=len(reader.seq))
                    for s in range(0, len(reader.seq)):
                        seq.append(
                            self.matcher.convert_ascii_to_encoding(
                                reader.seq[s]
                            )
                        )
                    var record = ByteFastaRecord(
                        List(reader.name.as_span()), seq
                    )
                    a_bytes_saved += record.size_in_bytes()
                    a_sequences.append(SeqAndIndex(record, a_seq_number))
                    a_seq_number += 1
                if a_bytes_saved >= self.settings.batch_size or not a_do_work:
                    return 0
            return 0

        @parameter
        fn a_process() capturing -> Int:
            print("a process")
            if a_bytes_saved >= self.settings.batch_size or not a_do_work:
                print("passed if?", a_bytes_saved, a_do_work, len(a_sequences))
                var outputs = List[Optional[ComputedMatchResult]](
                    capacity=len(a_sequences)
                )
                for _ in range(0, len(a_sequences)):
                    outputs.append(None)

                cpu_parallel_starts_ends[M, SeqAndIndex](
                    self.matcher,
                    self.settings,
                    a_sequences,
                    outputs,
                )
                print("did chunk")
                for i in range(0, len(outputs)):
                    var m = outputs[i]
                    if not m:
                        continue
                    var r = Pointer(to=a_sequences[i])

                    # Convert back to asii
                    for i in range(0, len(r[].seq.seq)):
                        r[].seq.seq[i] = self.matcher.convert_encoding_to_ascii(
                            r[].seq.seq[i]
                        )

                    writer.write(">")
                    writer.write_bytes(r[].seq.name)
                    writer.write("\n")

                    if (
                        self.settings.tty_info.is_a_tty
                        and self.settings.is_output_stdout()
                    ):
                        writer.write_bytes(
                            r[].seq.seq[0 : m.value().result.start]
                        )
                        writer.write("\033[1;31m")
                        writer.write(RED)
                        writer.write_bytes(
                            r[].seq.seq[
                                m.value().result.start : m.value().result.end
                            ]
                        )
                        writer.write()
                        writer.write(RESET)
                        writer.write_bytes(r[].seq.seq[m.value().result.end :])
                    else:
                        writer.write_bytes(r[].seq.seq)
                    writer.write("\n")
                a_sequences.clear()
                a_bytes_saved = 0
                a_seq_number = 0
            return 0

        @parameter
        fn b_read() capturing -> Int:
            print("b read")
            while b_do_work:
                var ret: Int = -1
                try:
                    ret = reader.read()
                except:
                    Logger.error("Failed to read from file", String(file))
                    return -1

                if ret == 0:
                    b_do_work = False
                elif ret < 0:
                    return -1
                else:
                    var seq = List[UInt8](capacity=len(reader.seq))
                    for s in range(0, len(reader.seq)):
                        seq.append(
                            self.matcher.convert_ascii_to_encoding(
                                reader.seq[s]
                            )
                        )
                    var record = ByteFastaRecord(
                        List(reader.name.as_span()), seq
                    )
                    b_bytes_saved += record.size_in_bytes()
                    b_sequences.append(SeqAndIndex(record, a_seq_number))
                    b_seq_number += 1
                if b_bytes_saved >= self.settings.batch_size or not a_do_work:
                    return 0
            return -1

        @parameter
        fn b_process() capturing -> Int:
            print("B process")
            if b_bytes_saved >= self.settings.batch_size or not a_do_work:
                var outputs = List[Optional[ComputedMatchResult]](
                    capacity=len(b_sequences)
                )
                for _ in range(0, len(b_sequences)):
                    outputs.append(None)

                cpu_parallel_starts_ends[M, SeqAndIndex](
                    self.matcher,
                    self.settings,
                    b_sequences,
                    outputs,
                )
                for i in range(0, len(outputs)):
                    var m = outputs[i]
                    if not m:
                        continue
                    var r = Pointer(to=b_sequences[i])

                    # Convert back to asii
                    for i in range(0, len(r[].seq.seq)):
                        r[].seq.seq[i] = self.matcher.convert_encoding_to_ascii(
                            r[].seq.seq[i]
                        )

                    writer.write(">")
                    writer.write_bytes(r[].seq.name)
                    writer.write("\n")

                    if (
                        self.settings.tty_info.is_a_tty
                        and self.settings.is_output_stdout()
                    ):
                        writer.write_bytes(
                            r[].seq.seq[0 : m.value().result.start]
                        )
                        writer.write("\033[1;31m")
                        writer.write(RED)
                        writer.write_bytes(
                            r[].seq.seq[
                                m.value().result.start : m.value().result.end
                            ]
                        )
                        writer.write()
                        writer.write(RESET)
                        writer.write_bytes(r[].seq.seq[m.value().result.end :])
                    else:
                        writer.write_bytes(r[].seq.seq)
                    writer.write("\n")
                b_sequences.clear()
                b_bytes_saved = 0
                b_seq_number = 0
            return 0

        run[a_read, a_process, b_read, b_process]()

        writer.flush()
        writer.close()

    fn run_search_on_file[
        W: MovableWriter
    ](mut self, file: Path, mut writer: BufferedWriter[W]) raises:
        var reader = FastxReader[read_comment=False](
            BufferedReader(GZFile(String(file), "r"))
        )

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
                var seq = List[UInt8](capacity=len(reader.seq))
                for s in range(0, len(reader.seq)):
                    seq.append(
                        self.matcher.convert_ascii_to_encoding(reader.seq[s])
                    )
                var record = ByteFastaRecord(List(reader.name.as_span()), seq)
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
                    var r = Pointer(to=sequences[i])

                    # Convert back to asii
                    for i in range(0, len(r[].seq.seq)):
                        r[].seq.seq[i] = self.matcher.convert_encoding_to_ascii(
                            r[].seq.seq[i]
                        )

                    writer.write(">")
                    writer.write_bytes(r[].seq.name)
                    writer.write("\n")

                    if (
                        self.settings.tty_info.is_a_tty
                        and self.settings.is_output_stdout()
                    ):
                        writer.write_bytes(
                            r[].seq.seq[0 : m.value().result.start]
                        )
                        writer.write("\033[1;31m")
                        writer.write(RED)
                        writer.write_bytes(
                            r[].seq.seq[
                                m.value().result.start : m.value().result.end
                            ]
                        )
                        writer.write()
                        writer.write(RESET)
                        writer.write_bytes(r[].seq.seq[m.value().result.end :])
                    else:
                        writer.write_bytes(r[].seq.seq)
                    writer.write("\n")
                sequences.clear()
                bytes_saved = 0
                seq_number = 0
        writer.flush()
        writer.close()


@value
struct GpuParallelFastaSearchRunner[
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
            record_type = RecordType.FASTA, check_record_size=True
        ](files[0])
        if first_peek.is_binary:
            for i in range(1, len(files)):
                first_peek = peek_file[
                    record_type = RecordType.FASTA, check_record_size=True
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
                    record_type = RecordType.FASTA, check_record_size=False
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
        var file_start = perf_counter()
        var reader = FastxReader[read_comment=False](
            BufferedReader(GZFile(String(path), "r"))
        )

        fn write_match[
            mut: Bool, //, o: Origin[mut]
        ](
            r: Pointer[ByteFastaRecord, o], m: ComputedMatchResult
        ) capturing raises:
            writer.write(">")
            writer.write_bytes(r[].name)
            writer.write("\n")
            if (
                self.settings.tty_info.is_a_tty
                and self.settings.is_output_stdout()
            ):
                writer.write_bytes(r[].seq[0 : m.result.start])
                writer.write(RED)
                writer.write_bytes(r[].seq[m.result.start : m.result.end])
                writer.write(RESET)
                writer.write_bytes(r[].seq[m.result.end :])
            else:
                writer.write_bytes(r[].seq)
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
                var seq = List[UInt8](capacity=len(reader.seq))
                for s in range(0, len(reader.seq)):
                    seq.append(
                        self.matcher.convert_ascii_to_encoding(reader.seq[s])
                    )
                var record = ByteFastaRecord(List(reader.name.as_span()), seq)

                if len(reader.seq) > max_target_length:
                    cpu_sequences.append(
                        SeqAndIndex(
                            record,
                            seq_index,
                        )
                    )
                else:
                    bytes_saved += max_target_length
                    sequences.append(SeqAndIndex(record, seq_index))

            seq_index += 1
            if (
                bytes_saved
                >= self.settings.readable_batch_size() - max_target_length
                or not do_work
            ):
                var done_reading = perf_counter()
                Logger.timing("Time reading", done_reading - start)
                var outputs = gpu_parallel_starts_ends[
                    M,
                    SeqAndIndex,
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
                        var r = Pointer(to=sequences[m.value().index].seq)
                        # Convert back to asii
                        for i in range(0, len(r[].seq)):
                            r[].seq[i] = self.matcher.convert_encoding_to_ascii(
                                r[].seq[i]
                            )
                        write_match(r, m.value())
                    else:
                        var r = Pointer(to=cpu_sequences[m.value().index].seq)
                        # Convert back to asii
                        for i in range(0, len(r[].seq)):
                            r[].seq[i] = self.matcher.convert_encoding_to_ascii(
                                r[].seq[i]
                            )
                        write_match(r, m.value())

                Logger.timing("write done:", perf_counter() - write_start)
                cpu_sequences.clear()
                sequences.clear()
                bytes_saved = 0
                seq_index = 0
        writer.flush()
        writer.close()
        var file_end = perf_counter()
        Logger.timing("Time to process", path, file_end - file_start)
