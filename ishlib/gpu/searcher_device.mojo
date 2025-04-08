from gpu.host import DeviceContext, DeviceBuffer, DeviceFunction
from gpu.memory import AddressSpace
from math import ceildiv
from memory import UnsafePointer, stack_allocation, memcpy

from ishlib.matcher import Searchable
from ishlib.gpu.dynamic_2d_matrix import Dynamic2DMatrix, StorageFormat


@value
@register_passable
struct BlockInfo:
    var num_targets: UInt
    var query_len: UInt
    var matrix_len: UInt
    var max_target_length: UInt
    var targets_len: UInt


@value
struct SearcherDevice[func_type: AnyTrivialRegType, //, func: func_type]:
    var ctx: DeviceContext

    var block_info: Optional[BlockInfo]

    var host_query: Optional[DeviceBuffer[DType.uint8]]
    var host_targets: Optional[DeviceBuffer[DType.uint8]]
    var host_target_lengths: Optional[DeviceBuffer[DType.uint32]]
    var host_scoring_matrix: Optional[DeviceBuffer[DType.int8]]
    var host_scores: Optional[DeviceBuffer[DType.int32]]
    var host_query_ends: Optional[DeviceBuffer[DType.int32]]
    var host_target_ends: Optional[DeviceBuffer[DType.int32]]

    var device_query: Optional[DeviceBuffer[DType.uint8]]
    var device_targets: Optional[DeviceBuffer[DType.uint8]]
    var device_target_lengths: Optional[DeviceBuffer[DType.uint32]]
    var device_scoring_matrix: Optional[DeviceBuffer[DType.int8]]
    var device_scores: Optional[DeviceBuffer[DType.int32]]
    var device_query_ends: Optional[DeviceBuffer[DType.int32]]
    var device_target_ends: Optional[DeviceBuffer[DType.int32]]

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
    fn create_devices() raises -> List[Self]:
        var ret = List[Self]()
        for i in range(0, DeviceContext.number_of_devices()):
            var device = DeviceContext(i)
            # Triple check it's a gpu
            if device.api() == "cuda" or device.api() == "hip":
                ret.append(
                    Self(
                        device,
                    )
                )
        return ret

    fn set_block_info(
        mut self,
        num_targets: UInt,
        query_len: UInt,
        matrix_len: UInt,
        max_target_length: UInt = 1024,
    ):
        self.block_info = BlockInfo(
            num_targets,
            query_len,
            matrix_len,
            max_target_length,
            num_targets * max_target_length,
        )

    fn host_create_input_buffers(
        mut self,
    ) raises:
        self.host_query = self.ctx.enqueue_create_host_buffer[DType.uint8](
            self.block_info.value().query_len
        )
        self.host_scoring_matrix = self.ctx.enqueue_create_host_buffer[
            DType.int8
        ](self.block_info.value().matrix_len)
        self.host_targets = self.ctx.enqueue_create_host_buffer[DType.uint8](
            self.block_info.value().targets_len
        )
        self.host_target_lengths = self.ctx.enqueue_create_host_buffer[
            DType.uint32
        ](self.block_info.value().num_targets)

    fn device_create_input_buffers(
        mut self,
    ) raises:
        self.device_query = self.ctx.enqueue_create_buffer[DType.uint8](
            self.block_info.value().query_len
        )
        self.device_scoring_matrix = self.ctx.enqueue_create_buffer[DType.int8](
            self.block_info.value().matrix_len
        )
        self.device_targets = self.ctx.enqueue_create_buffer[DType.uint8](
            self.block_info.value().targets_len
        )
        self.device_target_lengths = self.ctx.enqueue_create_buffer[
            DType.uint32
        ](self.block_info.value().num_targets)

    fn host_create_output_buffers(mut self) raises:
        self.host_scores = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.block_info.value().num_targets
        )
        self.host_query_ends = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.block_info.value().num_targets
        )
        self.host_target_ends = self.ctx.enqueue_create_host_buffer[
            DType.int32
        ](self.block_info.value().num_targets)

    fn device_create_output_buffers(
        mut self,
    ) raises:
        self.device_scores = self.ctx.enqueue_create_buffer[DType.int32](
            self.block_info.value().num_targets
        )
        self.device_query_ends = self.ctx.enqueue_create_buffer[DType.int32](
            self.block_info.value().num_targets
        )
        self.device_target_ends = self.ctx.enqueue_create_buffer[DType.int32](
            self.block_info.value().num_targets
        )
        self.ctx.enqueue_memset(self.device_scores.value(), 0)
        self.ctx.enqueue_memset(self.device_query_ends.value(), 0)
        self.ctx.enqueue_memset(self.device_target_ends.value(), 0)

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
            self.host_query.value().unsafe_ptr(), query.unsafe_ptr(), len(query)
        )

        memcpy(
            self.host_scoring_matrix.value().unsafe_ptr(),
            score_matrix,
            score_matrix_len,
        )
        # Create interleaved target seqs
        var buffer = self.host_targets.value()
        var coords = Dynamic2DMatrix[StorageFormat.ColumnMajor](
            self.block_info.value().max_target_length, len(targets)
        )

        for i in range(0, len(targets)):
            for j in range(0, len(targets[i].buffer_to_search())):
                buffer[coords.cord2idx(j, i)] = targets[i].buffer_to_search()[j]
            self.host_target_lengths.value()[i] = len(
                targets[i].buffer_to_search()
            )

    fn copy_inputs_to_device(read self) raises:
        self.host_targets.value().enqueue_copy_to(self.device_targets.value())
        self.host_target_lengths.value().enqueue_copy_to(
            self.device_target_lengths.value()
        )
        self.host_query.value().enqueue_copy_to(self.device_query.value())
        self.host_scoring_matrix.value().enqueue_copy_to(
            self.device_scoring_matrix.value()
        )

    fn copy_outputs_to_host(read self) raises:
        self.device_scores.value().enqueue_copy_to(self.host_scores.value())
        self.device_query_ends.value().enqueue_copy_to(
            self.host_query_ends.value()
        )
        self.device_target_ends.value().enqueue_copy_to(
            self.host_target_ends.value()
        )

    fn launch_kernel(mut self, threads_to_launch: UInt = 15000) raises:
        self._process_with_coarse_graining(
            threads_to_launch=threads_to_launch,
        )

    fn get_score(read self, idx: UInt) -> Int32:
        return self.host_scores.value()[idx]

    fn get_qeury_end(read self, idx: UInt) -> Int32:
        return self.host_query_ends.value()[idx]

    fn get_target_end(read self, idx: UInt) -> Int32:
        return self.host_target_ends.value()[idx]

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
            self.block_info.value().query_len,
            self.block_info.value().num_targets,
            threads_to_launch,
            grid_dim=num_blocks,
            block_dim=block_size,
        )
        # ctx.synchronize()
