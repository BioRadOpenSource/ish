from sys.info import has_amd_gpu_accelerator, has_nvidia_gpu_accelerator


@always_inline
fn has_gpu() -> Bool:
    return has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
