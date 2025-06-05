from sys.info import sizeof

from ishlib.gpu.searcher_device import Searchable


@value
struct ByteFastxRecord(Searchable):
    var name: List[UInt8]
    var seq: List[UInt8]
    var qual: Optional[List[UInt8]]

    fn __init__(
        out self,
        name: List[UInt8],
        seq: List[UInt8],
        qual: Optional[List[UInt8]] = None,
    ):
        self.name = name
        self.seq = seq
        self.qual = qual

    fn size_in_bytes(read self) -> UInt:
        return sizeof[Self]() + len(self.name) + len(self.seq)

    fn buffer_to_search(ref self) -> Span[UInt8, __origin_of(self)]:
        # This lifetime of seq is at least as long as self
        return rebind[Span[UInt8, __origin_of(self)]](Span(self.seq))
