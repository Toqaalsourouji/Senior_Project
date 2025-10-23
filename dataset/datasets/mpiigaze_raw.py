import os
import glob
import math
import logging
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    from scipy.io import loadmat
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

dic = {}
dicImage = {}

def _find_subject_days(normalized_root: str) -> List[Tuple[str, List[str]]]:
    """
    Return list of (subject_dir, [day_mat_files...]) under Normalized/.
    Matches files like day01.mat, day02.mat, etc. (case-insensitive),
    directly under pXX or one level deeper.
    """
    # subjects: p00..p14 (case-insensitive)
    subjects = sorted([p for p in glob.glob(os.path.join(normalized_root, "[pP][0-9][0-9]"))
                       if os.path.isdir(p)])
    out = []

    for s in subjects:
        # match dayNN.mat directly under pXX
        mats = sorted(glob.glob(os.path.join(s, "[dD][aA][yY][0-9][0-9].mat")))
        # also allow one-level nesting: pXX/*/dayNN.mat
        mats += sorted(glob.glob(os.path.join(s, "*", "[dD][aA][yY][0-9][0-9].mat")))
        # de-duplicate while preserving order
        seen = set(); mats_uniq = []
        for m in mats:
            if m not in seen:
                mats_uniq.append(m); seen.add(m)

        if mats_uniq:
            out.append((s, mats_uniq))

    return out



def _safe_load_image(path: str):
    # Many MPIIGaze normalized images are stored as .jpg/.png after MATLAB export.
    # If your normalized data are still in .mat with image arrays, we handle that too below.
    return Image.open(path).convert("RGB")


class MPIIGazeRaw(Dataset):
    """
    Minimal Dataset that walks MPIIGaze/Data/Normalized/pXX/dayYY and yields:
      (image_tensor, binned_labels, regression_labels, meta)
    to match what your train loop expects.

    We assume each 'day' dir contains either:
      - a set of image files (*.jpg, *.png) PLUS a parallel .mat or .npy file with angles, or
      - a single MATLAB file per frame (containing image & angles).
    We try common patterns and skip anything we cannot parse.
    """

    def __init__(self, data_dir: str, bins: int, binwidth: float, angle: float, transform=None):
        """
        Args:
          data_dir: path that contains MPIIGaze/Data/ (we will look under Data/Normalized/)
          bins, binwidth, angle: come from config.data_config['mpiigaze'] (your script sets these)
          transform: optional torchvision transform (resize/normalize, etc.) applied to the image
        """
        super().__init__()
        self.transform = transform
        self.bins = bins
        self.binwidth = binwidth
        self.angle = angle

        self.samples = []  # list of (img_path_or_mat, is_mat, pitch_deg, yaw_deg)

        normalized_root = os.path.join("MPIIGaze", "MPIIGaze", "Data", "Normalized")
        
        if not os.path.isdir(normalized_root):
            # also try if user pointed data_dir directly at .../MPIIGaze/Data
            alt = os.path.join(data_dir, "Data", "Normalized")
            if os.path.isdir(alt):
                normalized_root = alt
            else:
                raise FileNotFoundError(f"Could not find Normalized/ under {normalized_root} or {alt}")

        subj_days = _find_subject_days(normalized_root)
        if not subj_days:
            logging.warning("No pXX/dayYY found under Normalized/. Is your path correct?")

        # Heuristics:
        # 1) If we find .mat files with fields for image+gaze per frame -> read directly.
        # 2) Else, if we find image files AND a sidecar 'gaze.mat' or 'labels.mat' -> map by index.
        # 3) Else, try '*.npy' with a dict {'pitch': [...], 'yaw': [...]}.

        for subj_dir, day_dirs in subj_days:
            for day_dir in day_dirs:
                if day_dir.lower().endswith(".mat"):
                    if not _HAVE_SCIPY:
                        raise RuntimeError("SciPy is required to read .mat files. Run: pip install scipy")
                    # Use the right eye by default; change to 'left' if you want the left eye
                    self._add_mat_file(day_dir, eye="right")
                else:
                    continue
        #         # Case A: per-frame .mat files
        #         mat_frames = sorted(glob.glob(os.path.join(day_dir)))
        #         print(mat_frames)
        #         img_frames = sorted(glob.glob(os.path.join(day_dir, "*.jpg"))) + \
        #                      sorted(glob.glob(os.path.join(day_dir, "*.png")))
                

        #         if mat_frames and _HAVE_SCIPY:
        #             for m in mat_frames:
        #                 self.samples.append((m, True, None, None))
        #             continue

        #         # Case B: image files + a single labels file
        #         if img_frames:
        #             label_mat = None
        #             for cand in ["gaze.mat", "labels.mat", "label.mat"]:
        #                 p = os.path.join(day_dir, cand)
        #                 if os.path.isfile(p):
        #                     label_mat = p
        #                     break

        #             if label_mat and _HAVE_SCIPY:
        #                 md = loadmat(label_mat)
        #                 # Try common keys
        #                 pitch_list = None
        #                 yaw_list = None
        #                 for k in ["pitch", "Pitch", "gaze_pitch", "gazePitch"]:
        #                     if k in md:
        #                         pitch_list = np.squeeze(md[k])
        #                         break
        #                 for k in ["yaw", "Yaw", "gaze_yaw", "gazeYaw"]:
        #                     if k in md:
        #                         yaw_list = np.squeeze(md[k])
        #                         break

        #                 if pitch_list is not None and yaw_list is not None:
        #                     n = min(len(img_frames), len(pitch_list), len(yaw_list))
        #                     for i in range(n):
        #                         self.samples.append((img_frames[i], False, float(pitch_list[i]), float(yaw_list[i])))
        #                     continue

        #             # Case C: image files + numpy labels
        #             npy = os.path.join(day_dir, "labels.npy")
        #             if os.path.isfile(npy):
        #                 arr = np.load(npy, allow_pickle=True).item()
        #                 pitch_list = arr.get("pitch", [])
        #                 yaw_list = arr.get("yaw", [])
        #                 n = min(len(img_frames), len(pitch_list), len(yaw_list))
        #                 for i in range(n):
        #                     self.samples.append((img_frames[i], False, float(pitch_list[i]), float(yaw_list[i])))
        #                 continue

        if len(self.samples) == 0:
            raise RuntimeError("Found no usable frames/labels under Normalized/. "
                               "If your data are only in MATLAB .mat, please install scipy (pip install scipy).")

    def __len__(self):
        return len(self.samples)

    def _mat_to_image_and_angles(self, mat_path):
        md = loadmat(mat_path)
        # Try the most common field names
        # - image data can be in keys like 'image', 'face', etc. (uint8 HxWx3)
        # - angles can be degrees in keys like 'gaze', 'pitch','yaw' (we normalize below)
        img = None
        for k in ["image", "img", "face", "Face", "data"]:
            if k in md:
                img_arr = md[k]
                # MATLAB arrays can come transposed; try to coerce to HxWx3
                img_np = np.array(img_arr)
                if img_np.ndim == 3:
                    img = Image.fromarray(img_np.astype(np.uint8))
                    break

        pitch_deg = None
        yaw_deg = None

        # If 'gaze' provided as [pitch, yaw] in degrees
        for gk in ["gaze", "gaze_angle", "gazeAngles"]:
            if gk in md:
                gv = np.squeeze(md[gk])
                if gv.shape[-1] == 2 or gv.shape[0] == 2:
                    pitch_deg = float(gv[0])
                    yaw_deg   = float(gv[1])
                    break

        # Or separate keys
        if pitch_deg is None:
            for pk in ["pitch", "Pitch", "gaze_pitch"]:
                if pk in md:
                    pitch_deg = float(np.squeeze(md[pk]))
                    break
        if yaw_deg is None:
            for yk in ["yaw", "Yaw", "gaze_yaw"]:
                if yk in md:
                    yaw_deg = float(np.squeeze(md[yk]))
                    break

        return img, pitch_deg, yaw_deg

    def _bin_angles(self, pitch_deg: float, yaw_deg: float):
        # Convert continuous degrees -> class bins expected by your training loop
        # mapping: class_index * binwidth - angle  ~  [-angle, +angle]
        # Inverse: class_index = round((deg + angle)/binwidth)
        def to_bin(a):
            idx = int(round((a + self.angle) / self.binwidth))
            idx = max(0, min(self.bins - 1, idx))
            return idx
        return to_bin(pitch_deg), to_bin(yaw_deg)
    """"
    def __getitem__(self, idx):
        path, is_mat, pitch_deg, yaw_deg = self.samples[idx]

        if is_mat:
            if not _HAVE_SCIPY:
                raise RuntimeError("scipy not available to read .mat files")
            img, p_deg, y_deg = self._mat_to_image_and_angles(path)
            if img is None or p_deg is None or y_deg is None:
                raise RuntimeError(f"Could not parse {path}")
        else:
            img = _safe_load_image(path)
            p_deg = pitch_deg
            y_deg = yaw_deg

        if self.transform is not None:
            img = self.transform(img)

        # classification labels (bins)
        label_pitch_cls, label_yaw_cls = self._bin_angles(p_deg, y_deg)

        # regression labels (continuous deg)
        reg = torch.tensor([p_deg, y_deg], dtype=torch.float32)

        # Return in the same 4-tuple structure your loop expects:
        # (image_tensor, labels_gaze, regression_labels_gaze, meta)
        labels_gaze = torch.tensor([label_pitch_cls, label_yaw_cls], dtype=torch.long)
        meta = {"src": path, "pitch_deg": p_deg, "yaw_deg": y_deg}
        return img, labels_gaze, reg, meta
        """
    def __getitem__(self, idx):
        sample_info, is_mat, pitch_deg, yaw_deg = self.samples[idx]
        
        # sample_info is a tuple like ("MAT", mat_path, frame_index)
        sample_type, mat_path, frame_index = sample_info
        
        if sample_type == "MAT":
            # For MAT files, load from cached arrays
            if not hasattr(self, "_mat_cache") or mat_path not in self._mat_cache:
                raise RuntimeError(f"MAT cache missing for {mat_path}")
            
            cache = self._mat_cache[mat_path]
            
            # Get image from cached array
            img_array = cache["images"][frame_index]  # Shape: (H, W) for grayscale
            
            # Convert numpy array to PIL Image
            if img_array.ndim == 2:  # Grayscale
                img = Image.fromarray(img_array, mode='L').convert("RGB")
            elif img_array.ndim == 3:  # RGB
                img = Image.fromarray(img_array, mode='RGB')
            else:
                raise ValueError(f"Unexpected image array shape: {img_array.shape}")
            
            # Get angles from cache (they should match the function parameters)
            p_deg = float(cache["pitch"][frame_index])
            y_deg = float(cache["yaw"][frame_index])
            
        else:
            # For regular image files (if you have any)
            if isinstance(sample_info, tuple):
                # This shouldn't happen now, but just in case
                raise ValueError(f"Unexpected sample format: {sample_info}")
            
            img = _safe_load_image(sample_info)  # sample_info should be a path string
            p_deg = pitch_deg
            y_deg = yaw_deg

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

        # Create classification labels (bins)
        label_pitch_cls, label_yaw_cls = self._bin_angles(p_deg, y_deg)

        # Create regression labels (continuous degrees)
        reg = torch.tensor([p_deg, y_deg], dtype=torch.float32)

        # Create final labels
        labels_gaze = torch.tensor([label_pitch_cls, label_yaw_cls], dtype=torch.long)
        
        # Create metadata
        meta = {
            "src": mat_path, 
            "frame": frame_index,
            "pitch_deg": p_deg, 
            "yaw_deg": y_deg
        }
        
        return img, labels_gaze, reg, meta



    
    def _add_mat_file(self, mat_path: str, eye: str = "right"):
        """
        Index all frames in a dayNNN.mat aggregate file with robust structure detection.
        """
        print(f"Loading MAT file: {mat_path} for eye: {eye}")
        
        # Try different loading options
        try:
            # First attempt: load with squeeze_me=True (original approach)
            md = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        except:
            try:
                # Second attempt: load without squeezing
                md = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
            except:
                # Third attempt: load as record arrays
                md = loadmat(mat_path, squeeze_me=True, struct_as_record=True)
        
        if "data" not in md:
            raise KeyError(f"{mat_path}: no 'data' key found. Available keys: {list(md.keys())}")
        
        data = md["data"]

        
        # Handle different data wrapper formats
        if isinstance(data, np.ndarray):
            if data.ndim == 0:  # 0-dimensional array wrapper
                data = data.item()
            elif data.ndim == 1 and len(data) == 1:  # 1-element array
                data = data[0]
            else:
                # Try to find the actual data structure
                if hasattr(data, 'dtype') and data.dtype.names:
                    # Structured array
                    data = data[0] if data.ndim > 0 else data
        
        
        # Get eye data
        eye_data = None
        
        # Try different ways to access eye data
        if hasattr(data, eye):
            eye_data = getattr(data, eye)
        elif isinstance(data, dict) and eye in data:
            eye_data = data[eye]
        elif hasattr(data, 'dtype') and data.dtype.names and eye in data.dtype.names:
            eye_data = data[eye]
        else:
            # List all available attributes/keys for debugging
            if hasattr(data, '__dict__'):
                available = list(data.__dict__.keys())
            elif isinstance(data, dict):
                available = list(data.keys())
            elif hasattr(data, 'dtype') and data.dtype.names:
                available = list(data.dtype.names)
            else:
                available = [attr for attr in dir(data) if not attr.startswith('_')]
            
            raise KeyError(f"{mat_path}: eye '{eye}' not found. Available: {available}")
        
        
        # Handle eye_data being wrapped in arrays
        if isinstance(eye_data, np.ndarray):
            if eye_data.ndim == 0:
                eye_data = eye_data.item()
            elif eye_data.ndim == 1 and len(eye_data) == 1:
                eye_data = eye_data[0]
        
        # Extract gaze and image data
        gaze_data = None
        image_data = None
        
        # Try different ways to access gaze and image
        for attr_name in ['gaze', 'Gaze', 'gaze_data']:
            if hasattr(eye_data, attr_name):
                gaze_data = getattr(eye_data, attr_name)
                break
            elif isinstance(eye_data, dict) and attr_name in eye_data:
                gaze_data = eye_data[attr_name]
                break
            elif hasattr(eye_data, 'dtype') and eye_data.dtype.names and attr_name in eye_data.dtype.names:
                gaze_data = eye_data[attr_name]
                break
        
        for attr_name in ['image', 'Image', 'img', 'face', 'Face']:
            if hasattr(eye_data, attr_name):
                image_data = getattr(eye_data, attr_name)
                break
            elif isinstance(eye_data, dict) and attr_name in eye_data:
                image_data = eye_data[attr_name]
                break
            elif hasattr(eye_data, 'dtype') and eye_data.dtype.names and attr_name in eye_data.dtype.names:
                image_data = eye_data[attr_name]
                break
        
        if gaze_data is None or image_data is None:
            # Debug: show what's available in eye_data
            if hasattr(eye_data, '__dict__'):
                available = list(eye_data.__dict__.keys())
            elif isinstance(eye_data, dict):
                available = list(eye_data.keys())
            elif hasattr(eye_data, 'dtype') and eye_data.dtype.names:
                available = list(eye_data.dtype.names)
            else:
                available = [attr for attr in dir(eye_data) if not attr.startswith('_')]
            
            raise KeyError(f"{mat_path}: Could not find gaze/image data. Available in {eye} eye: {available}")
        
        
        # Process gaze data
        pitch_deg, yaw_deg = self._vecs_to_pitch_yaw_deg(gaze_data)
        n_gaze = len(pitch_deg)
        
        # Process image data
        I = np.asarray(image_data)
        
        # Handle different image formats
        if I.ndim == 2:
            I = I[None, ...]  # (H, W) -> (1, H, W)
        elif I.ndim == 3:
            if I.shape[-1] in [1, 3, 4] and I.shape[0] > 4:  # (H, W, C) format
                I = I[None, ...]  # (H, W, C) -> (1, H, W, C)
            # else assume (N, H, W) format
        elif I.ndim == 4:
            if I.shape[0] == 1 and I.shape[-1] > 10:  # Likely (1, H, W, N)
                I = np.moveaxis(I, -1, 0)  # (1, H, W, N) -> (N, H, W, 1)
                I = I.squeeze(-1)  # (N, H, W, 1) -> (N, H, W)
        
        # Convert to grayscale uint8 if needed
        if I.ndim == 4:
            if I.shape[-1] == 3:  # RGB
                I = np.mean(I, axis=-1).astype(np.uint8)
            elif I.shape[-1] == 1:  # Grayscale with channel
                I = I.squeeze(-1).astype(np.uint8)
        else:
            I = I.astype(np.uint8)
        
        n_images = I.shape[0]
        
        # Match lengths
        n_total = min(n_gaze, n_images)
        if n_gaze != n_images:
            pitch_deg = pitch_deg[:n_total]
            yaw_deg = yaw_deg[:n_total]
            I = I[:n_total]
        
        
        # Cache the processed data
        if not hasattr(self, "_mat_cache"):
            self._mat_cache = {}
        self._mat_cache[mat_path] = {
            "images": I, 
            "pitch": pitch_deg, 
            "yaw": yaw_deg
        }
        

        # Add to global dictionaries and samples list
        if mat_path not in dic:
            dic[mat_path] = []
            dicImage[mat_path] = []
        
        for k in range(n_total):
            sample = (("MAT", mat_path, k), True, float(pitch_deg[k]), float(yaw_deg[k])) #changed the false to true because all samples are from mat files
            dic[mat_path].append(sample)
            mainPathPart = sample[0][1].split("/Normalized/")[0]
            personDayPathPart = sample[0][1].split("/Normalized/")[1].split(".mat")[0]  # Added the trailing slash
            Opth = "Original"
            path = os.path.join(f"{mainPathPart}", Opth, personDayPathPart, f"{sample[0][2]:04d}.jpg")
            dicImage[mat_path].append(path)
            self.samples.append(sample)
        

        print(f"Successfully added {n_total} samples from {mat_path}")
        print(f"Total samples in dataset: {len(self.samples)}")
        return n_total


    def _vecs_to_pitch_yaw_deg(self, G):

        g = np.asarray(G)
        print(f"Input gaze - type: {type(G)}, array shape: {g.shape}, dtype: {g.dtype}")
        
        # Handle object arrays (common in MATLAB exports)
        if g.dtype == object:
            vectors = []
            # Flatten and process each element
            for item in g.flat:
                vec = np.asarray(item, dtype=np.float32)
                if vec.size == 3:
                    vectors.append(vec.reshape(3))
                elif vec.ndim == 2 and vec.shape[0] == 1 and vec.shape[1] == 3:
                    vectors.append(vec.reshape(3))
                elif vec.ndim == 2 and vec.shape[0] == 3 and vec.shape[1] == 1:
                    vectors.append(vec.reshape(3))
            
            if not vectors:
                raise ValueError("No valid 3D gaze vectors found in object array")
            
            g = np.vstack(vectors)
            print(f"Converted {len(vectors)} object vectors to shape: {g.shape}")
        
        # Handle regular numeric arrays
        elif g.ndim == 1 and g.size == 3:
            g = g.reshape(1, 3).astype(np.float32)
        elif g.ndim == 2:
            if g.shape[1] == 3:
                g = g.astype(np.float32)  # (N, 3)
            elif g.shape[0] == 3:
                g = g.T.astype(np.float32)  # (3, N) -> (N, 3)
            else:
                raise ValueError(f"2D gaze array shape {g.shape} not recognized")
        elif g.ndim == 3:
            # Handle cases like (N, 3, 1) or (N, 1, 3)
            if 3 in g.shape:
                # Reshape to (N, 3)
                if g.shape[1] == 3:
                    g = g.squeeze().astype(np.float32)
                elif g.shape[2] == 3:
                    g = g.squeeze().astype(np.float32)
                elif g.shape[0] == 3:
                    g = g.squeeze().T.astype(np.float32)
            else:
                raise ValueError(f"3D gaze array shape {g.shape} not recognized")
        else:
            # Try reshaping if divisible by 3
            if g.size % 3 == 0:
                n_vectors = g.size // 3
                g = g.reshape(n_vectors, 3).astype(np.float32)
            else:
                raise ValueError(f"Cannot handle gaze data with shape {g.shape}")
        
        # Ensure we have (N, 3) shape
        if g.ndim != 2 or g.shape[1] != 3:
            raise ValueError(f"Final gaze shape {g.shape} is not (N, 3)")
        
        # Convert to pitch/yaw
        x, y, z = g[:, 0], g[:, 1], g[:, 2]
        yaw = np.arctan2(x, -z)
        pitch = np.arctan2(y, np.sqrt(x * x + z * z))
        
        pitch_deg = np.degrees(pitch).astype(np.float32)
        yaw_deg = np.degrees(yaw).astype(np.float32)
        
        print(f"Converted to {len(pitch_deg)} pitch/yaw pairs")
        return pitch_deg, yaw_deg