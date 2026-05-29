import torch
import sys
import numpy as np

def print_structure(data, name="root", indent=0):
    spacing = "  " * indent
    prefix = f"{spacing} {name}: "

    # Handle PyTorch Tensors
    if torch.is_tensor(data):
        content_snippet = ""
        if data.numel() > 0:
            # Get a small snippet of the data
            flattened = data.flatten()
            snippet_vals = flattened[:3].tolist()
            content_snippet = f" | Content: {snippet_vals}..." if data.numel() > 3 else f" | Content: {snippet_vals}"
        
        print(f"{prefix}Tensor [Shape: {list(data.shape)} | Dtype: {data.dtype}]{content_snippet}")

    # Handle Dictionaries
    elif isinstance(data, dict):
        print(f"{prefix}Dict ({len(data)} keys)")
        for key, value in data.items():
            print_structure(value, name=str(key), indent=indent + 1)

    # Handle Lists or Tuples
    elif isinstance(data, (list, tuple)):
        print(f"{prefix}{type(data).__name__} (Length: {len(data)})")
        for i, item in enumerate(data):
            # To avoid flooding the terminal with long lists, you can limit this
            if i < 10:
                print_structure(item, name=f"Index {i}", indent=indent + 1)
            else:
                print(f"{spacing}   ... (truncated)")
                break

    # Handle Basic Values (Strings, Ints, etc.)
    else:
        val_str = str(data)
        if len(val_str) > 50:
            val_str = val_str[:47] + "..."
        print(f"{prefix}{type(data).__name__} = {val_str}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python read_pt.py <path_to_file.pt>")
        return

    path = sys.argv[1]
    try:
        # map_location='cpu' ensures it loads even if saved on GPU
        data = torch.load(path, map_location='cpu', weights_only=False)
        print(f"\nStructure of: {path}")
        print("-" * 40)
        print_structure(data)
        print("-" * 40)
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    main()