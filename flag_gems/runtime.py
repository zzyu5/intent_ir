import contextlib
import torch


class TorchDeviceFn:
    @contextlib.contextmanager
    def device(self, dev):
        if dev is None:
            yield
        else:
            with torch.cuda.device(dev):
                yield


torch_device_fn = TorchDeviceFn()

__all__ = ["torch_device_fn"]
