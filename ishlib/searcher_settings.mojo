from collections import Optional

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
                "min_score",
                OptKind.IntLike,
                default_value=String("1"),
                description="The min score needed to return a match.",
            )
        )
        parser.add_opt(
            OptConfig(
                "match_algo",
                OptKind.StringLike,
                default_value=String("naive_exact"),
                description=(
                    "The algorithm to use for matching: [naive_exact, ssw,"
                    " sw_local]"
                ),
            )
        )
        parser.add_opt(
            OptConfig(
                "record_type",
                OptKind.StringLike,
                default_value=String("line"),
                description="The input record type: [line, fasta]",
            )
        )

        try:
            var opts = parser.parse_sys_args()
            if opts.get_bool("help"):
                print(opts.get_help_message()[])
                return None

            var pattern = List(opts.get_string("pattern").as_bytes())
            var min_score = opts.get_int("min_score")
            var match_algo = opts.get_string("match_algo")
            var record_type = opts.get_string("record_type")
            var files = opts.args
            if len(files) == 0:
                print("missing files")
                raise "Expected files, found none."

            return Self(files, pattern, min_score, match_algo, record_type)
        except e:
            print(parser.help_msg())
            print(e)
            return None
