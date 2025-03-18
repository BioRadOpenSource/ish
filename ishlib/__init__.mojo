from memory import Span


struct ByteSpanWriter[origin: ImmutableOrigin](Writable):
    var inner: Span[UInt8, origin]

    fn __init__(out self, b: Span[UInt8, origin]):
        self.inner = b

    fn write_to[W: Writer](read self, mut writer: W):
        writer.write_bytes(self.inner)
