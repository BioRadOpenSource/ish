from memory import Span


# Terminal colors
alias RED = "\x1b[31m"
alias GREEN = "\x1b[32m"
alias YELLOW = "\x1b[33m"
alias BLUE = "\x1b[34m"
alias RESET = "\x1b[0m"


struct ByteSpanWriter[origin: ImmutableOrigin](Writable):
    var inner: Span[UInt8, origin]

    fn __init__(out self, b: Span[UInt8, origin]):
        self.inner = b

    fn write_to[W: Writer](read self, mut writer: W):
        writer.write_bytes(self.inner)


@value
@register_passable
struct RecordType:
    var value: Int
    alias LINE = Self(0)
    alias FASTA = Self(1)

    fn __eq__(read self, other: Self) -> Bool:
        return self.value == other.value
