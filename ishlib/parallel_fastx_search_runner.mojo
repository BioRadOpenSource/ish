from ExtraMojo.io.buffered import BufferedWriter

from time.time import perf_counter


from ishlib import RED, PURPLE, GREEN, RESET
from ishlib.formats.fastx import ByteFastxRecord
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
from ishlib.peek_file import peek_file, PeekFindings
from ishlib.matcher.alignment.striped_utils import AlignmentResult
from ishlib.vendor.kseq import BufferedReader, FastxReader
from ishlib.vendor.zlib import GZFile
from ishlib.vendor.log import Logger

from algorithm.functional import parallelize
from math import ceildiv
from pathlib import Path
from sys import stdout, info


@value
struct SeqAndIndex(SearchableWithIndex):
    var seq: ByteFastxRecord
    var orig_index: UInt

    fn buffer_to_search(ref self) -> Span[UInt8, __origin_of(self)]:
        return rebind[Span[UInt8, __origin_of(self)]](
            self.seq.buffer_to_search()
        )

    fn original_index(read self) -> UInt:
        return self.orig_index


@value
struct ParallelFastxSearchRunner[M: Matcher]:
    var settings: SearcherSettings
    var matcher: M

    fn run_search[
        W: Movable & Writer
    ](mut self, mut writer: BufferedWriter[W]) raises:
        # Peek only the first file to determine fastq or not, assume not-binary
        var peek = peek_file[record_type = RecordType.FASTX](
            self.settings.files[0]
        )
        for file in self.settings.files:
            var f = file  # force copy
            Logger.debug("Processing", f)
            if peek.is_fastq:
                self.run_search_on_file[is_fastq=True](f, writer)
            else:
                self.run_search_on_file[is_fastq=False](f, writer)

    fn run_search_on_file[
        W: Movable & Writer, *, is_fastq: Bool = False
    ](mut self, file: Path, mut writer: BufferedWriter[W]) raises:
        # TODO: pass an enocoder to the FastaReader
        var reader = FastxReader[read_comment=False](
            BufferedReader(GZFile(String(file), "rb"))
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
                var record: ByteFastxRecord

                @parameter
                if is_fastq:
                    record = ByteFastxRecord(
                        List(reader.name.as_span()),
                        seq,
                        List(reader.qual.as_span()),
                    )
                else:
                    record = ByteFastxRecord(
                        List(reader.name.as_span()),
                        seq,
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

                    @parameter
                    if is_fastq:
                        writer.write("\n+\n")
                        writer.write_bytes(r[].seq.qual.value())

                    writer.write("\n")
                sequences.clear()
                bytes_saved = 0
                seq_number = 0
        writer.flush()
        writer.close()


@value
struct GpuParallelFastxSearchRunner[
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
        W: Movable & Writer
    ](mut self, mut writer: BufferedWriter[W]) raises:
        # Peek the first file to get the suggested size, then use that for all of them.
        # Assume non-binary
        var files = self.settings.files
        var first_peek = peek_file[
            record_type = RecordType.FASTX, check_record_size=True
        ](files[0])

        Logger.debug("Suggested length of:", first_peek.suggested_max_length)

        # Create ctxs
        alias MAX_TARGET_LENGTHS = List(128, 256, 512, 1024, 2048, 4096)

        @parameter
        for i in range(0, len(MAX_TARGET_LENGTHS)):
            alias max_target_length = MAX_TARGET_LENGTHS[i]
            if first_peek.suggested_max_length <= max_target_length:
                var ctxs = self.create_ctxs[
                    max_query_length, max_target_length
                ]()
                self.search_files[
                    W,
                    max_query_length=max_query_length,
                    max_target_length=max_target_length,
                ](files, ctxs, writer, first_peek)
                return
        else:
            Logger.warn(
                "Longer line lengths than supported, more work will"
                " be sent to CPU, consider running with max-gpus set to 0."
            )
            var ctxs = self.create_ctxs[max_query_length, 4096]()
            self.search_files[
                W, max_query_length=max_query_length, max_target_length=4096
            ](files, ctxs, writer, first_peek)

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
        W: Movable & Writer,
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
        read peek: PeekFindings,
    ) raises:
        for i in range(0, len(paths)):
            var f = paths[i]  # force copy
            Logger.debug("Processing", f)
            # Assume that since the user specified FASTX, there are no binary files
            # Assume that all files are the same record type
            if peek.is_fastq:
                self.run_search_on_file[
                    W, max_query_length, max_target_length, is_fastq=True
                ](f, ctxs, writer)
            else:
                self.run_search_on_file[
                    W, max_query_length, max_target_length, is_fastq=False
                ](f, ctxs, writer)

    fn run_search_on_file[
        W: Movable & Writer,
        max_query_length: UInt = 200,
        max_target_length: UInt = 1024,
        *,
        is_fastq: Bool = False,
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
            r: Pointer[ByteFastxRecord, o], m: ComputedMatchResult
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

            @parameter
            if is_fastq:
                writer.write("\n+\n")
                writer.write_bytes(r[].qual.value())
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
                var record: ByteFastxRecord

                @parameter
                if is_fastq:
                    record = ByteFastxRecord(
                        List(reader.name.as_span()),
                        seq,
                        List(reader.qual.as_span()),
                    )
                else:
                    record = ByteFastxRecord(List(reader.name.as_span()), seq)

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
