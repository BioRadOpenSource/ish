from extramojo.io.buffered import BufferedWriter

from ishlib import RED, PURPLE, GREEN, ByteSpanWriter, RecordType
from ishlib.matcher import Matcher
from ishlib.peek_file import peek_file
from ishlib.searcher_settings import SearcherSettings
from ishlib.vendor.kseq import FastxReader, BufferedReader
from ishlib.vendor.zlib import GZFile
from ishlib.vendor.log import Logger

from pathlib import Path
from sys import stdout


@fieldwise_init
struct FastxSearchRunner[M: Matcher]:
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
        var reader = FastxReader[read_comment=False](
            BufferedReader(GZFile(String(file), "r"))
        )

        # TODO: hold onto the non-newline stripped sequence as well for outputting the match color

        while True:
            var ret = reader.read()
            if ret <= 0:
                break
            var seq = List[UInt8](capacity=len(reader.seq))
            for s in range(0, len(reader.seq)):
                seq.append(
                    self.matcher.convert_ascii_to_encoding(reader.seq[s])
                )

            var m = self.matcher.first_match(seq, self.settings.pattern)
            if m:
                # Convert back to ascii - needed for uniformity of casing changes
                for i in range(0, len(seq)):
                    seq[i] = self.matcher.convert_encoding_to_ascii(seq[i])
                writer.write(">")
                writer.write_bytes(reader.name.as_span())
                writer.write("\n")
                if (
                    self.settings.tty_info.is_a_tty
                    and self.settings.is_output_stdout()
                ):
                    writer.write_bytes(Span(seq)[0 : m.value().start])
                    writer.write(RED)
                    writer.write_bytes(
                        Span(seq)[m.value().start : m.value().end]
                    )
                    writer.write()
                    writer.write(RESET)
                    writer.write_bytes(Span(seq)[m.value().end :])
                else:
                    writer.write_bytes(seq)

                # Handle FASTQ Case
                @parameter
                if is_fastq:
                    writer.write("\n+\n")
                    writer.write_bytes(reader.qual.as_span())
                writer.write("\n")

        writer.flush()
        writer.close()
