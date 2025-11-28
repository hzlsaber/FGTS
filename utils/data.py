"""
Data loading and processing utilities

Provides unified dataset classes and data loading functions
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import torchvision.transforms as transforms

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SimpleImageDataset(Dataset):
    """
    Simple binary classification image dataset (real/fake)

    Args:
        real_dir: Real image directory
        fake_dir: Fake image directory
        transform: Image preprocessing transform
        max_samples: Maximum samples per class (None means use all)
    """
    def __init__(self, real_dir, fake_dir, transform=None, max_samples=None):
        self.transform = transform

        real_dir = Path(real_dir)
        fake_dir = Path(fake_dir)

        # Gather image files
        real_imgs = self._gather_images(real_dir)
        fake_imgs = self._gather_images(fake_dir)

        # Limit sample count
        if max_samples:
            real_imgs = real_imgs[:max_samples]
            fake_imgs = fake_imgs[:max_samples]

        # Merge and create labels
        self.images = real_imgs + fake_imgs
        self.labels = [0] * len(real_imgs) + [1] * len(fake_imgs)

        print(f"[Dataset] real={len(real_imgs)}, fake={len(fake_imgs)}, total={len(self.images)}")

    def _gather_images(self, directory: Path) -> List[Path]:
        """Gather all image files in directory"""
        if not directory.exists():
            return []

        images = []
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.webp']
        for ext in extensions:
            images.extend(list(directory.glob(ext)))

        return sorted(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            # If image loading fails, print warning and return a black image
            print(f"[Warning] Failed to load image {img_path}: {e}")
            # Create a black placeholder image (assuming 224x224, adjust if needed)
            if self.transform:
                # Apply transform to get the correct output size
                dummy_img = Image.new('RGB', (224, 224), (0, 0, 0))
                img = self.transform(dummy_img)
            else:
                img = torch.zeros(3, 224, 224)
            return img, label


def build_dataloader(real_dir, fake_dir, transform,
                     batch_size=32, max_samples=None,
                     shuffle=False, num_workers=4) -> DataLoader:
    """
    Build DataLoader

    Args:
        real_dir: Real image directory
        fake_dir: Fake image directory
        transform: Image preprocessing
        batch_size: Batch size
        max_samples: Maximum samples per class
        shuffle: Whether to shuffle
        num_workers: Number of data loading processes

    Returns:
        DataLoader object, returns None if dataset is empty
    """
    dataset = SimpleImageDataset(real_dir, fake_dir, transform, max_samples)

    if len(dataset) == 0:
        return None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def get_all_test_datasets(test_base_dir: str, test_mode: str = 'so-fake-ood') -> List[Dict]:
    """
    Scan test directory and return information for all test datasets

    Args:
        test_base_dir: Test dataset base directory
        test_mode: Test mode - 'so-fake-ood', 'GenImage', or 'AIGCDetectionBenchmark'

    Returns:
        List of dataset information, each element contains:
        - name: Dataset name
        - path: Dataset path
        - has_categories: Whether it has category subdirectories
        - categories: Category list
    """
    test_base_dir = Path(test_base_dir)
    datasets = []

    if test_mode == 'GenImage':
        # GenImage mode: dataset/val/real and fake
        for dataset_dir in sorted(test_base_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            val_dir = dataset_dir / 'val'
            if val_dir.exists() and (val_dir / 'real').exists() and (val_dir / 'fake').exists():
                datasets.append({
                    'name': dataset_dir.name,
                    'path': val_dir,  # Point to val directory
                    'has_categories': False,
                    'categories': [None],
                })

    elif test_mode == 'AIGCDetectionBenchmark':
        # AIGCDetectionBenchmark mode: may have nested subdirectories
        def scan_for_datasets(base_dir: Path, prefix: str = ''):
            for subdir in sorted(base_dir.iterdir()):
                if not subdir.is_dir():
                    continue

                # Check if this directory has 0_real and 1_fake
                if (subdir / '0_real').exists() and (subdir / '1_fake').exists():
                    dataset_name = f"{prefix}{subdir.name}" if prefix else subdir.name
                    datasets.append({
                        'name': dataset_name,
                        'path': subdir,
                        'has_categories': False,
                        'categories': [None],
                    })
                else:
                    # Recursively scan subdirectories
                    new_prefix = f"{prefix}{subdir.name}/" if prefix else f"{subdir.name}/"
                    scan_for_datasets(subdir, new_prefix)

        scan_for_datasets(test_base_dir)

    else:
        # Default: so-fake-ood mode (original behavior)
        for dataset_dir in sorted(test_base_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            # Check if it has category structure
            if (dataset_dir / '0_real').exists() and (dataset_dir / '1_fake').exists():
                # No category structure
                datasets.append({
                    'name': dataset_dir.name,
                    'path': dataset_dir,
                    'has_categories': False,
                    'categories': [None],
                })
            else:
                # Try category structure
                categories = []
                for cat_dir in sorted(dataset_dir.iterdir()):
                    if not cat_dir.is_dir():
                        continue
                    if (cat_dir / '0_real').exists() and (cat_dir / '1_fake').exists():
                        categories.append(cat_dir.name)

                if categories:
                    datasets.append({
                        'name': dataset_dir.name,
                        'path': dataset_dir,
                        'has_categories': True,
                        'categories': categories,
                    })

    return datasets


def resolve_dataset_paths(dataset_path: Path, category: Optional[str] = None, test_mode: str = 'so-fake-ood') -> Tuple[Path, Path]:
    """
    Parse dataset paths and return (real_dir, fake_dir)

    Args:
        dataset_path: Dataset root directory
        category: Category name (if dataset has category structure)
        test_mode: Test mode - affects directory naming expectations

    Returns:
        (real_dir, fake_dir) tuple
    """
    dataset_path = Path(dataset_path)

    if category:
        # Has category structure
        real_dir = dataset_path / category / '0_real'
        fake_dir = dataset_path / category / '1_fake'
    else:
        # No category structure - check for different naming conventions
        # First try standard naming (0_real/1_fake)
        real_dir = dataset_path / '0_real'
        fake_dir = dataset_path / '1_fake'

        # If not found, try GenImage naming (real/fake)
        if not real_dir.exists():
            if (dataset_path / 'real').exists():
                real_dir = dataset_path / 'real'
            if (dataset_path / 'fake').exists():
                fake_dir = dataset_path / 'fake'

    # Check other possible naming for fake directory
    if not fake_dir.exists():
        for name in ['1_fake_ldm', '1_fake_sd14', '1_fake_progan', 'fake']:
            candidate = fake_dir.parent / name
            if candidate.exists():
                fake_dir = candidate
                break

    return real_dir, fake_dir


def create_dataset_readme(dataset_dir: Path):
    """
    Create README documentation file in dataset directory

    Args:
        dataset_dir: Dataset directory
    """
    readme_content = """# Dataset Directory Structure

Please organize your datasets as follows:

## For training:
```
datasets/train/
├── progan/
│   ├── car/
│   │   ├── 0_real/          # Real images
│   │   ├── 1_fake_ldm/      # Fake images (LDM)
│   │   └── 1_fake_sd14/     # Fake images (SD 1.4)
│   └── person/
│       └── ...
```

## For testing:
```
datasets/test/
├── progan/
│   └── car/
│       ├── 0_real/
│       └── 1_fake/
├── stylegan/
│   └── car/
│       └── ...
```

## Supported image formats:
- PNG, JPG, JPEG, WEBP

## Notes:
- Each `0_real` directory should contain real images
- Each `1_fake*` directory should contain generated/fake images
- You can have multiple fake directories with different suffixes (e.g., `1_fake_ldm`, `1_fake_sd14`)
"""

    readme_path = dataset_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"Created dataset README at {readme_path}")
