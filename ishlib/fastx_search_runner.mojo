from ExtraMojo.io import MovableWriter
from ExtraMojo.io.buffered import BufferedWriter

from ishlib import RED, PURPLE, GREEN, ByteSpanWriter, RecordType
from ishlib.matcher import Matcher
from ishlib.peek_file import peek_file
from ishlib.searcher_settings import SearcherSettings
from ishlib.vendor.kseq import FastxReader, BufferedReader
from ishlib.vendor.zlib import GZFile
from ishlib.vendor.log import Logger

from pathlib import Path
from sys import stdout


@value
struct FastxSearchRunner[M: Matcher]:
    var settings: SearcherSettings
    var matcher: M

    fn run_search[
        W: MovableWriter
    ](mut self, mut writer: BufferedWriter[W]) raises:
        # Simple thing first?
        for file in self.settings.files:
            var f = file[]  # force copy
            var peek = peek_file[record_type = RecordType.FASTX](f)
            if peek.is_binary:
                if self.settings.verbose:
                    Logger.warn("Skipping binary file:", file[])
                continue
            Logger.debug("Processing", f)
            self.run_search_on_file(f, writer)

    fn run_search_on_file[
        W: MovableWriter
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
                writer.write(">")
                writer.write_bytes(reader.name.as_span())
                writer.write("\n")
                if (
                    self.settings.tty_info.is_a_tty
                    and self.settings.is_output_stdout()
                ):
                    writer.write_bytes(
                        reader.seq.as_span()[0 : m.value().start]
                    )
                    writer.write(RED)
                    writer.write_bytes(
                        reader.seq.as_span()[m.value().start : m.value().end]
                    )
                    writer.write()
                    writer.write(RESET)
                    writer.write_bytes(reader.seq.as_span()[m.value().end :])
                else:
                    writer.write_bytes(reader.seq.as_span())

                # Handle FASTQ Case
                if len(reader.qual.as_span()) > 0:
                    writer.write("\n+\n")
                    writer.write_bytes(reader.qual.as_span())
                writer.write("\n")

        writer.flush()
        writer.close()
