"""Input size contract for CV exports. All stored sizes are [width, height]."""
INPUT_SIZE_ALIGNMENT = 32

_RESIZABLE_ARCHITECTURES = {
    "rockchip": {"yolov5", "yolov5_seg", "yolov6", "yolov7", "yolov8", "yolov8_obb",
                 "yolov8_pose", "yolov8_seg", "yolov10", "yolo11"},
    "sophon": {"yolov5", "yolov7", "yolov8", "yolov8_obb", "yolov8_pose", "yolov8_seg",
               "yolov10", "yolo11"},
}


def supports_custom_input_size(architecture, vendor):
    return architecture in _RESIZABLE_ARCHITECTURES.get(vendor, ())


def resolve_input_size(cfg, vendor, requested=None):
    support = cfg.get("vendor_support", {}).get(vendor, {})
    default = support.get("input_size") or cfg["input_size"]
    if requested is None:
        return list(default) if default else None
    if not support.get("custom_input_size"):
        if requested == default:
            return list(default)
        raise ValueError(f"{cfg['name']} 在当前芯片厂家下使用固定输入尺寸")
    if not isinstance(requested, (list, tuple)) or len(requested) != 2:
        raise ValueError("输入尺寸需要同时提供宽和高")

    dimensions = []
    for value in requested:
        if isinstance(value, str) and value.isascii() and value.isdecimal():
            try:
                value = int(value)
            except ValueError:
                raise ValueError("无法读取输入尺寸，请填写正整数像素值") from None
        if type(value) is not int or value <= 0:
            raise ValueError("输入宽和高必须是正整数")
        if value % INPUT_SIZE_ALIGNMENT:
            raise ValueError(f"输入宽和高都必须是 {INPUT_SIZE_ALIGNMENT} 的倍数")
        dimensions.append(value)
    return dimensions
