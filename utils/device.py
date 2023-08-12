import onnx
import onnxruntime

device_types_list = ["cpu", "cuda"]

available_providers = onnxruntime.get_available_providers()

def get_device_and_provider(device='cpu'):
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    if device == 'cuda':
        if "CUDAExecutionProvider" in available_providers:
            provider = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            device = 'cpu'
            provider = ["CPUExecutionProvider"]
    else:
        device = 'cpu'
        provider = ["CPUExecutionProvider"]

    return device, provider, options


data_type_bytes = {'uint8': 1, 'int8': 1, 'uint16': 2, 'int16': 2, 'float16': 2, 'float32': 4}


def estimate_max_batch_size(resolution, chunk_size=1024, data_type='float32', channels=3):
    pixel_size = data_type_bytes.get(data_type, 1)
    image_size = resolution[0] * resolution[1] * pixel_size * channels
    number_of_batches = (chunk_size * 1024 * 1024) // image_size
    return max(number_of_batches, 1)
