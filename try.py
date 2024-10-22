import torch

def test_cuda():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    test_cuda()
