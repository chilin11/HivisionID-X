import onnxruntime as ort
import os

def get_best_providers():
    """
    Returns a list of the best available ONNX Runtime execution providers
    based on the current hardware and installation.
    """
    available_providers = ort.get_available_providers()
    providers = []

    # Priority 1: CUDA (NVIDIA GPU)
    if 'CUDAExecutionProvider' in available_providers:
        providers.append('CUDAExecutionProvider')

    # Priority 2: TensorRT (NVIDIA GPU - optional, but very fast)
    if 'TensorrtExecutionProvider' in available_providers:
        providers.insert(0, 'TensorrtExecutionProvider')

    # Priority 3: OpenVINO (Intel / AMD CPUs and GPUs)
    if 'OpenVINOExecutionProvider' in available_providers:
        providers.append('OpenVINOExecutionProvider')

    # Fallback: CPU
    if 'CPUExecutionProvider' in available_providers:
        providers.append('CPUExecutionProvider')
    else:
        providers.append('CPUExecutionProvider') # Always include CPU fallback

    return providers

def load_onnx_model(checkpoint_path, set_cpu=False):
    """
    Loads an ONNX model safely using the best available execution providers.
    
    :param checkpoint_path: Path to the .onnx file.
    :param set_cpu: If True, forces the CPU provider.
    :return: An ONNX Runtime InferenceSession.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    providers = ['CPUExecutionProvider'] if set_cpu else get_best_providers()
    options = ort.SessionOptions()
    if os.getenv("RUN_MODE") != "beast":
        # Large encoder/decoder graphs can otherwise retain multi-GB arena
        # blocks beyond tensor lifetimes. Prefer bounded allocations by default;
        # explicit beast mode keeps ONNX Runtime's throughput-oriented defaults.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False

    try:
        sess = ort.InferenceSession(checkpoint_path, sess_options=options, providers=providers)
    except Exception as e:
        print(f"Failed to load model {checkpoint_path} with providers {providers}: {e}")
        print("Falling back to CPUExecutionProvider...")
        sess = ort.InferenceSession(checkpoint_path, sess_options=options, providers=['CPUExecutionProvider'])

    return sess
