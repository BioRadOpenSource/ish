from ExtraMojo.io.buffered import BufferedWriter

from ishlib import RED, PURPLE, GREEN, ByteSpanWriter, RecordType
from ishlib.matcher import Matcher
from ishlib.peek_file import peek_file
from ishlib.vendor.kseq import BufferedReader, SearchChar, ByteString
from ishlib.vendor.zlib import GZFile
from ishlib.vendor.log import Logger
from ishlib.searcher_settings import SearcherSettings

from pathlib import Path
from sys import stdout


@value
struct LineSearchRunner[M: Matcher]:
    var settings: SearcherSettings
    var matcher: M

    fn run_search[
        W: Movable & Writer
    ](mut self, mut writer: BufferedWriter[W]) raises:
        # Simple thing first?
        for file in self.settings.files:
            var f = file  # force copy
            var peek = peek_file[record_type = RecordType.LINE](f)
            if peek.is_binary:
                if self.settings.verbose:
                    Logger.warn("Skipping binary file:", file)
                continue
            Logger.debug("Processing", f)
            self.run_search_on_file(f, writer)

    fn run_search_on_file[
        W: Movable & Writer
    ](mut self, path: Path, mut writer: BufferedWriter[W]) raises:
        var file = String(path)
        var reader = BufferedReader(GZFile(file, "r"))
        var buffer = ByteString()
        var encoded = ByteString()

        var line_number = 1
        while True:
            buffer.clear()
            encoded.clear()
            if reader.read_until[SearchChar.Newline](buffer) < 0:
                break
            for i in range(0, len(buffer)):
                encoded.push(self.matcher.convert_ascii_to_encoding(buffer[i]))
            var m = self.matcher.first_match(
                encoded.as_span(), self.settings.pattern
            )
            if m:
                var b = buffer.as_span()
                if (
                    self.settings.tty_info.is_a_tty
                    and self.settings.is_output_stdout()
                ):
                    writer.write(PURPLE)
                    writer.write_bytes(file.as_bytes())
                    writer.write(RESET)
                    writer.write_bytes(":".as_bytes())
                    writer.write(GREEN)
                    writer.write(line_number)
                    writer.write(RESET)
                    writer.write(": ")
                    writer.write_bytes(b[0 : m.value().start])
                    writer.write(RED)
                    writer.write_bytes(b[m.value().start : m.value().end])
                    writer.write(RESET)
                    writer.write_bytes(b[m.value().end :])
                    writer.write("\n")
                else:
                    writer.write_bytes(b)
                    writer.write("\n")
            line_number += 1
        writer.flush()
        writer.close()
