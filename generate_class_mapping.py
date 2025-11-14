"""
Generate class_to_idx.json mapping from Database directory structure.

This script scans the Database directory and creates a JSON mapping file
that maps class names to indices for inference.

Usage:
    python generate_class_mapping.py
    
Output:
    models/class_to_idx.json
"""

import json
from pathlib import Path


def generate_class_mapping(data_dir: str = 'Database', output_path: str = 'models/class_to_idx.json'):
    """
    Generate class_to_idx.json from directory structure.
    
    Args:
        data_dir: Path to database directory with class subfolders
        output_path: Output path for class_to_idx.json
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Database directory not found: {data_dir}")
    
    # Get all class directories
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    
    if not classes:
        raise ValueError(f"No class directories found in {data_dir}")
    
    # Create class_to_idx mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    
    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    print(f"✅ Generated class mapping:")
    print(f"   Classes: {len(classes)}")
    print(f"   Output: {output_path}")
    print(f"\nClass mapping:")
    for class_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        print(f"   {idx:2d}: {class_name}")
    
    return class_to_idx


if __name__ == '__main__':
    generate_class_mapping()
