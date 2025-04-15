from collections import Optional
from sys.info import num_physical_cores

from ExtraMojo.cli.parser import OptParser, OptConfig, OptKind


@value
struct SearcherSettings:
    """Settings for the ish searcher."""

    var files: List[String]
    """The files to search for matches."""
    var pattern: List[UInt8]
    """The pattern to search for."""
    var min_score: Int
    """The minimum score needed to return a match."""

    var match_algo: String
    var record_type: String
    var threads: UInt
    var batch_size: UInt
    var no_gpu: Bool

    @staticmethod
    fn from_args() raises -> Optional[Self]:
        var parser = OptParser(
            name="ish", description="Search for inexact patterns in files."
        )
        parser.add_opt(
            OptConfig(
                "pattern",
                OptKind.StringLike,
                default_value=None,
                description="The pattern to search for.",
            )
        )
        parser.add_opt(
            OptConfig(
                "min-score",
                OptKind.IntLike,
                default_value=String("1"),
                description="The min score needed to return a match.",
            )
        )
        parser.add_opt(
            OptConfig(
                "match-algo",
                OptKind.StringLike,
                default_value=String("ssw"),
                description=(
                    "The algorithm to use for matching: [naive_exact,"
                    " striped-local, basic-local, basic-global,"
                    " basic-semi-global, striped-semi-global]"
                ),
            )
        )
        parser.add_opt(
            OptConfig(
                "record-type",
                OptKind.StringLike,
                default_value=String("line"),
                description="The input record type: [line, fasta]",
            )
        )
        parser.add_opt(
            OptConfig(
                "threads",
                OptKind.IntLike,
                default_value=String(num_physical_cores()),
                description=(
                    "The number of threads to use. Defaults to the number of"
                    " physical cores."
                ),
            )
        )
        parser.add_opt(
            OptConfig(
                "batch-size",
                OptKind.IntLike,
                default_value=String("10737418240"),
                # TODO: elaborate on this for GPU batch sizing, with multiple devices.
                description=(
                    "The number of bytes in a parallel processing batch. Note"
                    " that this may use 2-3x this amount to account for"
                    " intermediate transfer buffers. Default is 10GiB. Note"
                    " that this allows for overflow when parsing, records may"
                    " 'hang over' the end of this buffer amount if needed."
                ),
            )
        )
        parser.add_opt(
            OptConfig(
                "no-gpu",
                OptKind.BoolLike,
                is_flag=True,
                default_value=String("False"),
                description="Don't use the GPU(s), even if available.",
            )
        )

        parser.expect_at_least_n_args(
            1, "Files to search for the given pattern."
        )

        try:
            var opts = parser.parse_sys_args()
            if opts.get_bool("help"):
                print(opts.get_help_message()[])
                return None

            var pattern = List(opts.get_string("pattern").as_bytes())
            var min_score = opts.get_int("min-score")
            var match_algo = opts.get_string("match-algo")
            var record_type = opts.get_string("record-type")
            var threads = opts.get_int("threads")
            if threads <= 0:
                raise "Threads must be >= 1."
            var batch_size = opts.get_int("batch-size")
            if batch_size < 1024 * 10:
                raise "Batch size too small"
            var files = opts.args
            if len(files) == 0:
                print("missing files")
                raise "Expected files, found none."

            var no_gpu = opts.get_bool("no-gpu")

            return Self(
                files,
                pattern,
                min_score,
                match_algo,
                record_type,
                threads,
                batch_size,
                no_gpu,
            )
        except e:
            print(parser.help_msg())
            print(e)
            return None
