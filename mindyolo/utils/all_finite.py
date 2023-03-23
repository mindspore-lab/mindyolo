import mindspore as ms

# TODO: Delete this code when publish
def compare_version(v1, v2="2.0.0"):
    """
    :param v1: version, format like 1.8.1
    :param v2: version, format like 1.8.1
    :return: v1 </=/> v1, return -1/0/1
    """

    l1 = str(v1).split(".")
    l2 = str(v2).split(".")
    for i in range(min(len(l1), len(l2))):
        if int(l1[i]) == int(l2[i]):
            pass
        elif int(l1[i]) < int(l2[i]):
            return -1
        else:
            return 1
    if len(l1) == len(l2):
        return 0
    elif len(l1) < len(l2):
        return -1
    else:
        return 1

if compare_version(ms.__version__) < 0:
    from mindspore import context, ops

    _ascend_target = context.get_context("device_target") == "Ascend"
    _gpu_target = context.get_context("device_target") == "GPU"
    npu_alloc_float_status = ops.NPUAllocFloatStatus()
    npu_clear_float_status = ops.NPUClearFloatStatus()
    if context.get_context("device_target") == "Ascend":
        _status = npu_alloc_float_status()
        _ = npu_clear_float_status(_status)
    else:
        _status = None
    _hypermap = ops.HyperMap()
    _partial = ops.Partial()


    def _is_finite(inputs):
        if _gpu_target:
            return ops.FloatStatus()(inputs)[0] == 0
        status = ops.isfinite(inputs)
        return status.all()

    def all_finite(inputs, status=None):
        if _ascend_target:
            status = ops.depend(_status, inputs)
            get_status = ops.NPUGetFloatStatus()(status)
            status = ops.depend(status, get_status)
            status_finite = status.sum() == 0
            _ = ops.NPUClearFloatStatus()(status)
            return status_finite
        outputs = _hypermap(_partial(_is_finite), inputs)
        return ops.stack(outputs).all()
else:
    from mindspore.amp import all_finite
