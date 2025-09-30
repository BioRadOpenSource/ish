from memory import UnsafePointer
from sys.info import alignof
from gpu.host import DeviceBuffer


@fieldwise_init
struct StorageFormat:
    var value: Int
    alias ColumnMajor = Self(0)
    alias RowMajor = Self(1)

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value


struct Dynamic2DMatrix[
    storage_format: StorageFormat = StorageFormat.RowMajor,
]:
    var rows: UInt
    var cols: UInt

    # TODO: have this have a ref to underlying data
    fn __init__(
        out self,
        rows: UInt,
        cols: UInt,
    ):
        self.rows = rows
        self.cols = cols

    @always_inline
    fn cord2idx(read self, row: UInt, col: UInt) -> UInt:
        var idx: UInt

        @parameter
        if self.storage_format == StorageFormat.RowMajor:
            idx = row * self.cols + col
        else:
            idx = col * self.rows + row
        return idx

    # fn __getitem__(
    #     self, row: UInt, col: UInt
    # ) -> ref [origin, address_space] Self.T:
    #     return self.data[self.cord2idx(row, col)]


# fn main() raises:
#     var strings = List[List[UInt8]](
#         List("AAAAAAAAAAAAAAAA".as_bytes()),
#         List("CCCCCCCCCCCCCCCC".as_bytes()),
#         List("GGGGGGGGGGGGGGGG".as_bytes()),
#         List("TTTTTTTTTTTTTTTT".as_bytes()),
#     )

#     alias LEN = 16  # length of each string
#     alias COUNT = 4  # total number of strings

#     var buffer_a = List[UInt8](capacity=LEN * COUNT)
#     var buffer_b = List[UInt8](capacity=LEN * COUNT)
#     for _ in range(0, LEN * COUNT):
#         buffer_a.append(0)
#         buffer_b.append(0)
#     var custom = Dynamic2DStorage[
#         UInt8,
#         __origin_of(buffer_b),
#         storage_format = StorageFormat.ColumnMajor,
#     ](LEN, len(strings), buffer_b)

#     var tensor = LayoutTensor[DType.uint8, Layout.col_major(LEN, COUNT)](
#         buffer_a
#     )
#     for i in range(0, len(strings)):
#         for j in range(0, len(strings[i])):
#             tensor[j, i] = strings[i][j]
#             custom[j, i] = strings[i][j]

#     print(tensor)

#     var fmt = String()
#     for row in range(0, custom.rows):
#         for col in range(0, custom.cols):
#             fmt.write_bytes(String(custom[row, col]).as_bytes())
#             fmt.write_bytes(", ".as_bytes())
#         fmt.write_bytes("\n".as_bytes())
#     print(fmt)

#     for value in custom.data:
#         print(String(value[]), end=", ")
#     print()
