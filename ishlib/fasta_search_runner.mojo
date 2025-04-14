from ExtraMojo.io.buffered import BufferedWriter

from ishlib import ByteSpanWriter
from ishlib.matcher import Matcher
from ishlib.searcher_settings import SearcherSettings
from ishlib.vendor.kseq import FastxReader, BufferedReader
from ishlib.vendor.zlib import GZFile

from utils import StringSlice
from sys import stdout


@value
struct FastaSearchRunner[M: Matcher]:
    var settings: SearcherSettings
    var matcher: M

    fn run_search(mut self) raises:
        # Simple thing first?
        for file in self.settings.files:
            var f = file[]  # force copy
            self.run_search_on_file(f)

    fn run_search_on_file(mut self, file: String) raises:
        var reader = FastxReader[read_comment=False](
            BufferedReader(GZFile(file, "r"))
        )
        var writer = BufferedWriter(stdout)

        # TODO: hold onto the non-newline stripped sequence as well for outputting the match color

        while True:
            var ret = reader.read()
            if ret <= 0:
                break
            var m = self.matcher.first_match(
                reader.seq.as_span(), self.settings.pattern
            )
            if m:
                writer.write(">")
                writer.write_bytes(reader.name.as_span())
                writer.write("\n")
                writer.write_bytes(reader.seq.as_span()[0 : m.value().start])
                # writer.write("\033[1;31m")
                writer.write_bytes(
                    reader.seq.as_span()[m.value().start : m.value().end]
                )
                writer.write()
                # writer.write("\033[0m")
                writer.write_bytes(reader.seq.as_span()[m.value().end :])
                writer.write("\n")
        writer.flush()
        writer.close()
