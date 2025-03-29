from ExtraMojo.io.buffered import BufferedReader, BufferedWriter

from ishlib.formats.fasta import FastaReader, BorrowedFastaRecord
from ishlib.searcher_settings import SearcherSettings
from ishlib.matcher import Matcher
from ishlib import ByteSpanWriter

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
        var reader = FastaReader(BufferedReader(open(file, "r")))
        var writer = BufferedWriter(stdout)

        # TODO: hold onto the non-newline stripped sequence as well for outputting the match color

        while True:
            var record = reader.read_borrowed()
            if not record:
                break
            var m = self.matcher.first_match(
                record.value().seq, self.settings.pattern
            )
            if m:
                writer.write(">")
                writer.write_bytes(record.value().name)
                writer.write("\n")
                writer.write_bytes(record.value().seq[0 : m.value().start])
                writer.write("\033[1;31m")
                writer.write_bytes(
                    record.value().seq[m.value().start : m.value().end]
                )
                writer.write()
                writer.write("\033[0m")
                writer.write_bytes(record.value().seq[m.value().end :])
                writer.write("\n")
        writer.flush()
        writer.close()
