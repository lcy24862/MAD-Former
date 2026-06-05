import os
import csv
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from scipy.ndimage import zoom


class PETDataset(Dataset):
    """Dataset for ADNI PET data with CSV-based fold splits.

    CSV format: filename,mask,DX
    Supports arbitrary binary classification tasks (AD_HC, HC_MCI, EMCI_LMCI, etc.)
    """

    def __init__(self, csv_path, data_dir, label_map=None, target_size=(128, 128, 128)):
        """
        Args:
            csv_path:    Path to CSV file defining the fold split
            data_dir:    Base directory containing class subdirectories
                         (e.g. "data/18F-AV1451")
            label_map:   Dict mapping CSV 'DX' values to numeric labels.
                         e.g. {'AD': 1, 'HC': 0} or {'EMCI': 0, 'LMCI': 1}
                         If None, auto-detect from the CSV.
            target_size: Target 3D size to resize volumes to (D, H, W),
                         default (128,128,128) to match BFEN
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.label_map = label_map
        self.classes_ = None  # Will be set by _load_csv
        self.samples = self._load_csv(csv_path)

    def _extract_class_dir(self, csv_filename):
        """Extract class directory from CSV path like '../ADNI/18F-AV1451/AD/PET_xxx.nii.gz'."""
        parts = csv_filename.replace('\\', '/').split('/')
        # Path format: ../ADNI/<dataset>/<class_dir>/<filename>.nii.gz
        # The class_dir is the second-to-last component
        if len(parts) >= 2:
            return parts[-2]
        return None

    def _load_csv(self, csv_path):
        samples = []
        label_counts = {}

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract filename and class_dir from CSV path
                csv_filename = row['filename']
                filename = os.path.basename(csv_filename)
                class_dir = self._extract_class_dir(csv_filename)
                label_str = row['DX'].strip()

                # Determine numeric label
                if self.label_map is not None:
                    label = self.label_map.get(label_str)
                    if label is None:
                        print(f"[WARNING] Unknown label '{label_str}' for {filename}, skipping")
                        continue
                else:
                    # Auto-detect: first seen class = 0, second = 1
                    if label_str not in label_counts:
                        label_counts[label_str] = len(label_counts)
                    label = label_counts[label_str]

                # Store label counts for summary
                label_counts[label_str] = label_counts.get(label_str, 0) + 1

                # Construct local file path
                file_path = os.path.join(self.data_dir, class_dir, filename)

                if os.path.exists(file_path):
                    samples.append((file_path, label))
                else:
                    print(f"[WARNING] File not found, skipping: {file_path}")

        # Store class names for reporting
        if self.label_map:
            rev = {v: k for k, v in self.label_map.items()}
            self.classes_ = [rev.get(i, f'class_{i}') for i in range(len(self.label_map))]
        else:
            self.classes_ = sorted(label_counts.keys())

        print(f"Loaded {len(samples)} samples from {csv_path}")
        for cls_name in sorted(label_counts.keys()):
            print(f"  {cls_name}: {label_counts[cls_name]}")
        return samples

    def __getitem__(self, index):
        file_path, label = self.samples[index]

        data = np.array(nib.load(file_path).get_fdata(), dtype="float32")
        data = np.nan_to_num(data, neginf=0)

        data = self._normalize(data)
        data = self._resize(data, self.target_size)

        data = torch.tensor(data).unsqueeze(0)  # (1, D, H, W)
        return data, label

    def __len__(self):
        return len(self.samples)

    def _normalize(self, data):
        _range = np.max(data) - np.min(data)
        if _range > 0:
            return (data - np.min(data)) / _range
        return data

    def _resize(self, data, target_size):
        factors = [t / s for t, s in zip(target_size, data.shape)]
        return zoom(data, factors, order=1)

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
