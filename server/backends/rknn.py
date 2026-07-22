from pathlib import Path

from .base import ConversionBackend


class RknnBackend(ConversionBackend):
    name = "rknn"
    output_suffix = ".rknn"

    def convert(self, source, output, options, progress):
        if Path(source).suffix.lower() != ".onnx":
            raise ValueError("RKNN 后端当前需要 ONNX 中间模型")
        try:
            from rknn.api import RKNN
        except ImportError as exc:
            raise RuntimeError(
                "未安装 RKNN-Toolkit2。请把与 Ubuntu/Python 版本匹配的官方 wheel "
                "放入 wheels/ 后重新运行 scripts/setup_wsl.sh。"
            ) from exc

        progress(48, "初始化 RKNN-Toolkit2")
        rknn = RKNN(verbose=bool(options.get("verbose", False)))
        try:
            cfg = options["catalog"]
            config_options = {"target_platform": options["platform"]}
            # Image normalization is fused into RKNN.  Audio waveforms are
            # already normalized float32 tensors; passing a single mean for a
            # [1, samples] input makes Toolkit2 treat every sample as a channel.
            if cfg.get("modality", "image") == "image" and options.get("fuse_normalization", True):
                config_options.update(
                    mean_values=[cfg.get("mean", [0, 0, 0])],
                    std_values=[cfg.get("std", [255, 255, 255])],
                )
            ret = rknn.config(**config_options)
            if ret not in (None, 0):
                raise RuntimeError(f"rknn.config 失败，返回值 {ret}")
            progress(56, "加载 ONNX 图")
            load_options = {"model": str(source)}
            load_options.update(options.get("load_onnx", {}))
            ret = rknn.load_onnx(**load_options)
            if ret != 0:
                raise RuntimeError(f"rknn.load_onnx 失败，返回值 {ret}")
            quant = options["precision"] in ("i8", "u8")
            dataset = options.get("dataset") if quant else None
            if quant and not dataset:
                raise ValueError("INT8/UINT8 量化必须上传校准图片 ZIP")
            progress(65, "构建并优化 RKNN（此阶段通常耗时最长）")
            ret = rknn.build(do_quantization=quant, dataset=dataset)
            if ret != 0:
                raise RuntimeError(f"rknn.build 失败，返回值 {ret}")
            progress(86, "导出 RKNN 模型")
            ret = rknn.export_rknn(str(output))
            if ret != 0:
                raise RuntimeError(f"rknn.export_rknn 失败，返回值 {ret}")
        finally:
            rknn.release()
