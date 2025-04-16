from bit import next_power_of_two
from gpu.host import DeviceContext, DeviceBuffer, DeviceFunction
from gpu.memory import AddressSpace
from math import ceildiv
from memory import UnsafePointer, stack_allocation, memcpy

from ishlib.matcher import Searchable
from ishlib.matcher.alignment.scoring_matrix import MatrixKind
from ishlib.gpu.dynamic_2d_matrix import Dynamic2DMatrix, StorageFormat
from ishlib.vendor.log import Logger


@value
@register_passable
struct BlockInfo:
    var num_targets: UInt
    var query_len: UInt
    var matrix_len: UInt
    var max_target_length: UInt
    var targets_len: UInt
    var matrix_kind: MatrixKind


@value
struct DeviceBufferWrapper[location: StringLiteral, dtype: DType]:
    var buffer: DeviceBuffer[dtype]
    var size: UInt
    var capacity: UInt

    fn __init__(
        out self,
        ctx: DeviceContext,
        *,
        size: UInt,
        capacity: Optional[UInt] = None,
    ) raises:
        var cap = UInt(
            next_power_of_two(size)
        ) if not capacity else capacity.value()
        if location == "host":
            self.buffer = ctx.enqueue_create_host_buffer[dtype](cap)
        else:
            self.buffer = ctx.enqueue_create_buffer[dtype](cap)
        self.capacity = cap
        self.size = size

    fn resize(mut self, ctx: DeviceContext, *, new_size: UInt) raises:
        if new_size <= self.capacity:
            self.size = new_size
            return

        self.capacity = next_power_of_two(new_size)
        if location == "host":
            self.buffer = ctx.enqueue_create_host_buffer[dtype](self.capacity)
        else:
            self.buffer = ctx.enqueue_create_buffer[dtype](self.capacity)
        self.size = new_size

    fn as_span(ref self) -> Span[Scalar[dtype], __origin_of(self)]:
        return Span[Scalar[dtype], __origin_of(self)](
            ptr=self.buffer.unsafe_ptr(), length=self.size
        )


@value
struct SearcherDevice[func_type: AnyTrivialRegType, //, func: func_type]:
    var ctx: DeviceContext

    var block_info: Optional[BlockInfo]

    var host_query: Optional[DeviceBufferWrapper["host", DType.uint8]]
    var host_targets: Optional[DeviceBufferWrapper["host", DType.uint8]]
    var host_target_lengths: Optional[DeviceBufferWrapper["host", DType.uint32]]
    var host_scoring_matrix: Optional[DeviceBufferWrapper["host", DType.int8]]
    var host_scores: Optional[DeviceBufferWrapper["host", DType.int32]]
    var host_query_ends: Optional[DeviceBufferWrapper["host", DType.int32]]
    var host_target_ends: Optional[DeviceBufferWrapper["host", DType.int32]]

    var device_query: Optional[DeviceBufferWrapper["device", DType.uint8]]
    var device_targets: Optional[DeviceBufferWrapper["device", DType.uint8]]
    var device_target_lengths: Optional[
        DeviceBufferWrapper["device", DType.uint32]
    ]
    var device_scoring_matrix: Optional[
        DeviceBufferWrapper["device", DType.int8]
    ]
    var device_scores: Optional[DeviceBufferWrapper["device", DType.int32]]
    var device_query_ends: Optional[DeviceBufferWrapper["device", DType.int32]]
    var device_target_ends: Optional[DeviceBufferWrapper["device", DType.int32]]

    fn __init__(
        out self,
        owned ctx: DeviceContext,
    ):
        self.ctx = ctx^

        self.block_info = None

        # Host input buffers
        self.host_query = None
        self.host_targets = None
        self.host_target_lengths = None
        self.host_scoring_matrix = None

        # Host output buffers
        self.host_scores = None
        self.host_query_ends = None
        self.host_target_ends = None

        # Device input buffers
        self.device_query = None
        self.device_targets = None
        self.device_target_lengths = None
        self.device_scoring_matrix = None

        # Device output buffers
        self.device_scores = None
        self.device_query_ends = None
        self.device_target_ends = None

    @staticmethod
    fn create_devices(
        batch_size: UInt,
        query_length: UInt,
        matrix_length: UInt,
        *,
        max_target_length: UInt,
    ) raises -> List[Self]:
        var ret = List[Self]()
        for i in range(0, DeviceContext.number_of_devices()):
            var device = DeviceContext(i)
            # Triple check it's a gpu
            if device.api() == "cuda" or device.api() == "hip":
                var s = Self(device)
                s._create_some_input_buffers(
                    query_length, matrix_length, batch_size, max_target_length
                )
                ret.append(s)
            break

        return ret

    fn set_block_info(
        mut self,
        num_targets: UInt,
        query_len: UInt,
        matrix_len: UInt,
        matrix_kind: MatrixKind,
        max_target_length: UInt = 1024,
    ):
        self.block_info = BlockInfo(
            num_targets,
            query_len,
            matrix_len,
            max_target_length,
            max_target_length * num_targets,
            matrix_kind,
        )

    fn _create_some_input_buffers(
        mut self,
        query_len: UInt,
        matrix_len: UInt,
        targets_len: UInt,
        max_target_length: UInt,
    ) raises:
        var num_targets = ceildiv(targets_len, max_target_length)
        self.host_query = DeviceBufferWrapper["host", DType.uint8](
            self.ctx, size=query_len
        )
        self.host_query.value().resize(self.ctx, new_size=0)

        self.host_scoring_matrix = DeviceBufferWrapper["host", DType.int8](
            self.ctx, size=matrix_len
        )
        self.host_scoring_matrix.value().resize(self.ctx, new_size=0)

        self.host_targets = DeviceBufferWrapper["host", DType.uint8](
            self.ctx, size=targets_len
        )
        self.host_targets.value().resize(self.ctx, new_size=0)

        self.host_target_lengths = DeviceBufferWrapper["host", DType.uint32](
            self.ctx, size=num_targets
        )
        self.host_target_lengths.value().resize(self.ctx, new_size=0)

        # Create dev equivalents
        self.device_query = DeviceBufferWrapper["device", DType.uint8](
            self.ctx, size=query_len
        )
        self.device_query.value().resize(self.ctx, new_size=0)

        self.device_scoring_matrix = DeviceBufferWrapper["device", DType.int8](
            self.ctx, size=matrix_len
        )
        self.device_scoring_matrix.value().resize(self.ctx, new_size=0)

        self.device_targets = DeviceBufferWrapper["device", DType.uint8](
            self.ctx, size=targets_len
        )
        self.device_targets.value().resize(self.ctx, new_size=0)

        self.device_target_lengths = DeviceBufferWrapper[
            "device", DType.uint32
        ](self.ctx, size=num_targets)
        self.device_target_lengths.value().resize(self.ctx, new_size=0)

    fn host_create_input_buffers(
        mut self,
    ) raises:
        if self.host_query:
            self.host_query.value().resize(
                self.ctx, new_size=self.block_info.value().query_len
            )
        else:
            self.host_query = DeviceBufferWrapper["host", DType.uint8](
                self.ctx, size=self.block_info.value().query_len
            )

        if self.host_scoring_matrix:
            self.host_scoring_matrix.value().resize(
                self.ctx, new_size=self.block_info.value().matrix_len
            )
        else:
            self.host_scoring_matrix = DeviceBufferWrapper["host", DType.int8](
                self.ctx, size=self.block_info.value().matrix_len
            )

        if self.host_targets:
            self.host_targets.value().resize(
                self.ctx, new_size=self.block_info.value().targets_len
            )
        else:
            self.host_targets = DeviceBufferWrapper["host", DType.uint8](
                self.ctx, size=self.block_info.value().targets_len
            )

        if self.host_target_lengths:
            self.host_target_lengths.value().resize(
                self.ctx, new_size=self.block_info.value().num_targets
            )
        else:
            self.host_target_lengths = DeviceBufferWrapper[
                "host", DType.uint32
            ](self.ctx, size=self.block_info.value().num_targets)

    fn device_create_input_buffers(
        mut self,
    ) raises:
        if self.device_query:
            self.device_query.value().resize(
                self.ctx, new_size=self.block_info.value().query_len
            )
        else:
            if not self.host_query:
                raise "Host query buffer must be created before device query buffer."
            self.device_query = DeviceBufferWrapper["device", DType.uint8](
                self.ctx,
                size=self.host_query.value().size,
                capacity=self.host_query.value().capacity,
            )

        if self.device_scoring_matrix:
            self.device_scoring_matrix.value().resize(
                self.ctx, new_size=self.block_info.value().matrix_len
            )
        else:
            if not self.host_scoring_matrix:
                raise "Host scoring buffer must be created before device scoring buffer."
            self.device_scoring_matrix = DeviceBufferWrapper[
                "device", DType.int8
            ](
                self.ctx,
                size=self.host_scoring_matrix.value().size,
                capacity=self.host_scoring_matrix.value().capacity,
            )

        if self.device_targets:
            self.device_targets.value().resize(
                self.ctx, new_size=self.block_info.value().targets_len
            )
        else:
            if not self.host_targets:
                raise "Host targets buffer must be created before device targets buffer."
            self.device_targets = DeviceBufferWrapper["device", DType.uint8](
                self.ctx,
                size=self.host_targets.value().size,
                capacity=self.host_targets.value().capacity,
            )

        if self.device_target_lengths:
            self.device_target_lengths.value().resize(
                self.ctx, new_size=self.block_info.value().num_targets
            )
        else:
            if not self.host_target_lengths:
                raise "Host target_lengths buffer must be created before device target_lengths buffer."
            self.device_target_lengths = DeviceBufferWrapper[
                "device", DType.uint32
            ](
                self.ctx,
                size=self.host_target_lengths.value().size,
                capacity=self.host_target_lengths.value().capacity,
            )

    fn device_create_output_buffers(mut self) raises:
        if self.device_scores:
            self.device_scores.value().resize(
                self.ctx, new_size=self.block_info.value().matrix_len
            )
        else:
            self.device_scores = DeviceBufferWrapper["device", DType.int32](
                self.ctx, size=self.block_info.value().num_targets
            )

        if self.device_query_ends:
            self.device_query_ends.value().resize(
                self.ctx, new_size=self.block_info.value().matrix_len
            )
        else:
            self.device_query_ends = DeviceBufferWrapper["device", DType.int32](
                self.ctx, size=self.block_info.value().num_targets
            )

        if self.device_target_ends:
            self.device_target_ends.value().resize(
                self.ctx, new_size=self.block_info.value().matrix_len
            )
        else:
            self.device_target_ends = DeviceBufferWrapper[
                "device", DType.int32
            ](self.ctx, size=self.block_info.value().num_targets)

        self.ctx.enqueue_memset(self.device_scores.value().buffer, 0)
        self.ctx.enqueue_memset(self.device_query_ends.value().buffer, 0)
        self.ctx.enqueue_memset(self.device_target_ends.value().buffer, 0)

    fn host_create_output_buffers(
        mut self,
    ) raises:
        if self.host_scores:
            self.host_scores.value().resize(
                self.ctx, new_size=self.block_info.value().matrix_len
            )
        else:
            if not self.device_scores:
                raise "device scores buffer must be created before device scores buffer."
            self.host_scores = DeviceBufferWrapper["host", DType.int32](
                self.ctx,
                size=self.device_scores.value().size,
                capacity=self.device_scores.value().capacity,
            )

        if self.host_query_ends:
            self.host_query_ends.value().resize(
                self.ctx, new_size=self.block_info.value().matrix_len
            )
        else:
            if not self.device_query_ends:
                raise "device query_ends buffer must be created before device query_ends buffer."
            self.host_query_ends = DeviceBufferWrapper["host", DType.int32](
                self.ctx,
                size=self.device_query_ends.value().size,
                capacity=self.device_scores.value().capacity,
            )

        if self.host_target_ends:
            self.host_target_ends.value().resize(
                self.ctx, new_size=self.block_info.value().matrix_len
            )
        else:
            if not self.device_target_ends:
                raise "device target_ends buffer must be created before device target_ends buffer."
            self.host_target_ends = DeviceBufferWrapper["host", DType.int32](
                self.ctx,
                size=self.device_target_ends.value().size,
                capacity=self.device_scores.value().capacity,
            )

    fn synchronize(read self) raises:
        self.ctx.synchronize()

    fn set_host_inputs[
        S: Searchable
    ](
        mut self,
        query: Span[UInt8],
        read score_matrix: UnsafePointer[Int8],
        score_matrix_len: UInt,
        targets: Span[S],
    ):
        memcpy(
            self.host_query.value().buffer.unsafe_ptr(),
            query.unsafe_ptr(),
            len(query),
        )

        memcpy(
            self.host_scoring_matrix.value().buffer.unsafe_ptr(),
            score_matrix,
            score_matrix_len,
        )
        # Create interleaved target seqs
        var buffer = self.host_targets.value().buffer
        var coords = Dynamic2DMatrix[StorageFormat.ColumnMajor](
            self.block_info.value().max_target_length, len(targets)
        )

        for i in range(0, len(targets)):
            for j in range(0, len(targets[i].buffer_to_search())):
                buffer[coords.cord2idx(j, i)] = targets[i].buffer_to_search()[j]
            self.host_target_lengths.value().buffer[i] = len(
                targets[i].buffer_to_search()
            )

    fn copy_inputs_to_device(read self) raises:
        self.host_targets.value().buffer.enqueue_copy_to(
            self.device_targets.value().buffer
        )
        self.host_target_lengths.value().buffer.enqueue_copy_to(
            self.device_target_lengths.value().buffer
        )
        self.host_query.value().buffer.enqueue_copy_to(
            self.device_query.value().buffer
        )
        self.host_scoring_matrix.value().buffer.enqueue_copy_to(
            self.device_scoring_matrix.value().buffer
        )

    fn copy_outputs_to_host(read self) raises:
        self.device_scores.value().buffer.enqueue_copy_to(
            self.host_scores.value().buffer
        )
        self.device_query_ends.value().buffer.enqueue_copy_to(
            self.host_query_ends.value().buffer
        )
        self.device_target_ends.value().buffer.enqueue_copy_to(
            self.host_target_ends.value().buffer
        )

    fn launch_kernel(mut self, threads_to_launch: UInt = 15000) raises:
        self._process_with_coarse_graining(
            threads_to_launch=threads_to_launch,
        )

    fn get_score(read self, idx: UInt) -> Int32:
        return self.host_scores.value().buffer[idx]

    fn get_qeury_end(read self, idx: UInt) -> Int32:
        return self.host_query_ends.value().buffer[idx]

    fn get_target_end(read self, idx: UInt) -> Int32:
        return self.host_target_ends.value().buffer[idx]

    fn _process_with_coarse_graining[
        block_size: UInt = 32
    ](mut self, threads_to_launch: Int = 15000,) raises:
        var num_blocks = ceildiv(threads_to_launch, block_size)
        var aligner = self.ctx.compile_function[func]()

        self.ctx.enqueue_function(
            aligner,
            self.device_query.value(),
            self.device_targets.value(),
            self.device_target_lengths.value(),
            self.device_scores.value(),
            self.device_query_ends.value(),
            self.device_target_ends.value(),
            self.device_scoring_matrix.value(),
            self.block_info.value().matrix_len,
            self.block_info.value().matrix_kind,
            self.block_info.value().query_len,
            self.block_info.value().num_targets,
            threads_to_launch,
            grid_dim=num_blocks,
            block_dim=block_size,
        )
