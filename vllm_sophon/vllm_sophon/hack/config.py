import torch
from vllm.config import DeviceConfig

"""
device_type is 'sophtpu' in platform in sophtpu
if set device_type 'tpu', with conflict with Google tpu
in vllm v0.7.3, has ray_only_device=['tpu'], but not set in latest version
so just use device_type = 'sophtpu' and wrapper in v0.7.3, can del after
"""
def DeviceConfig__init__(self, device: str = "auto") -> None:
    if device == "auto":
        # Automated device type detection
        from vllm.platforms import current_platform
        self.device_type = current_platform.device_type
        if not self.device_type:
            raise RuntimeError("Failed to infer device type")
    else:
        # Device type is assigned explicitly
        self.device_type = device

    # Some device types require processing inputs on CPU
    if self.device_type in ["neuron", "openvino"]:
        self.device = torch.device("cpu")
    elif self.device_type in ["tpu"]:
        self.device = None
    elif self.device_type in ["sophtpu"]:
        self.device = torch.device(f"tpu:0")
    else:
        # Set device with device type
        self.device = torch.device(self.device_type)

DeviceConfig.__init__ = DeviceConfig__init__
