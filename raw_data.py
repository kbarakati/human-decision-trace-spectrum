"""Read original AFM/EELS files with the same conventions as the SV-DKL notebook.
No pre-exported pkl dataset is needed.
"""
from pathlib import Path
import numpy as np
import h5py
from SciFiReaders import IgorIBWReader


def load_nonlinearity(image_path, h5_path):
    image_path, h5_path = Path(image_path), Path(h5_path)
    dataset = IgorIBWReader(str(image_path)).read()
    image = np.asarray(dataset['Channel_001'].compute()).T.copy()
    with h5py.File(h5_path, 'r') as f:
        x_map = np.asarray(f['Image/XLVDT'][:])
        y_map = np.asarray(f['Image/YLVDT'][:])
        if x_map.shape != y_map.shape:
            raise ValueError('XLVDT and YLVDT shapes differ')
        grid_shape = x_map.shape
        # Check the first and last force-curve IDs before trusting indexing.
        for i, j in [(0, 0), (grid_shape[0]-1, grid_shape[1]-1)]:
            key = f'ForceMap/0/{i}:{j}'
            if key not in f:
                raise KeyError(f'Missing AFM curve {key}')
    n = int(np.prod(grid_shape))
    return {
        'dataset': 'nonlinearity', 'image': image, 'grid_shape': grid_shape,
        'coordinates': np.column_stack([x_map.ravel(), y_map.ravel()]),
        'ids': np.asarray([f'{i}:{j}' for i in range(grid_shape[0]) for j in range(grid_shape[1])]),
        'n': n, 'h5_path': str(h5_path), 'image_path': str(image_path),
        'axis_label': 'Bias (channel 7)', 'response_label': 'AFM amplitude (channel 2)',
    }


def load_eels(npy_path, dataset_id=0):
    npy_path = Path(npy_path)
    collection = np.load(npy_path, allow_pickle=True).item()
    key = dataset_id if dataset_id in collection else str(dataset_id)
    if key not in collection:
        raise KeyError(f'EELS dataset {dataset_id!r} not found; available: {list(collection)}')
    data = collection[key]
    image = np.asarray(data['image'])
    cube = np.asarray(data['spectrum image'])
    energy = np.asarray(data['energy axis'])
    if cube.ndim != 3 or image.shape != cube.shape[:2] or energy.shape != (cube.shape[2],):
        raise ValueError(f'EELS shapes do not align: image={image.shape}, cube={cube.shape}, energy={energy.shape}')
    nx, ny, _ = cube.shape
    return {
        'dataset': 'eels', 'image': image, 'grid_shape': (nx, ny),
        'coordinates': np.asarray([(i, j) for i in range(nx) for j in range(ny)]),
        'ids': np.asarray([f'{i}:{j}' for i in range(nx) for j in range(ny)]),
        'n': nx*ny, 'cube': cube, 'energy': energy, 'npy_path': str(npy_path),
        'dataset_id': str(key), 'axis_label': 'Energy (eV)', 'response_label': 'EELS intensity',
    }


def make_patches(image, grid_shape, indices, patch_size=16):
    # Matches prepare_data(): row/column centers scaled from spectral grid to image.
    nx, ny = grid_shape
    h, w = image.shape
    half = patch_size // 2
    before, after = half, patch_size-half
    padded = np.pad(image, ((before, after), (before, after)), mode='reflect')
    out = []
    for idx in indices:
        i, j = divmod(int(idx), ny)
        row = int((i + 0.5) * h / nx)
        col = int((j + 0.5) * w / ny)
        row = min(max(row, 0), h-1)
        col = min(max(col, 0), w-1)
        out.append(np.asarray(padded[row:row+patch_size, col:col+patch_size], dtype=np.float32))
    return np.stack(out)


def get_response(data, idx):
    idx = int(idx)
    nx, ny = data['grid_shape']
    i, j = divmod(idx, ny)
    if data['dataset'] == 'eels':
        x = np.asarray(data['energy'], dtype=np.float32)
        response = np.asarray(data['cube'][i, j], dtype=np.float32)
    else:
        with h5py.File(data['h5_path'], 'r') as f:
            curve = f[f'ForceMap/0/{i}:{j}'][:]
        if curve.ndim != 2 or curve.shape[0] < 8:
            raise ValueError(f'Unexpected AFM ForceMap curve shape: {curve.shape}')
        x = np.asarray(curve[7], dtype=np.float32)
        response = np.asarray(curve[2], dtype=np.float32)
    mask = np.isfinite(x) & np.isfinite(response)
    return x[mask], response[mask]


def schedule(data, n_candidates=20, n_steps=100, seed=42, patch_size=16):
    if n_candidates < 2:
        raise ValueError('n_candidates must be >=2')
    max_steps = min(int(n_steps), data['n']//n_candidates)
    if max_steps < 1:
        raise ValueError('Not enough measurement positions')
    order = np.random.default_rng(seed).permutation(data['n'])[:max_steps*n_candidates]
    patches = make_patches(data['image'], data['grid_shape'], order, patch_size)
    finite = patches[np.isfinite(patches)]
    vmin, vmax = np.percentile(finite, [1, 99]) if len(finite) else (0., 1.)
    return {
        'n_candidates': n_candidates, 'n_steps': max_steps, 'seed': seed,
        'indices': order.tolist(), 'ids': data['ids'][order].tolist(),
        'coordinates': data['coordinates'][order].tolist(), 'patches': patches,
        'vmin': float(vmin), 'vmax': float(vmax),
        'dataset': data['dataset'], 'axis_label': data['axis_label'],
        'response_label': data['response_label'],
    }


def get_full_response(data, idx):
    """Return the measured response and full original measurement at a location."""
    idx = int(idx)
    nx, ny = data["grid_shape"]
    i, j = divmod(idx, ny)
    if data["dataset"] == "eels":
        axis = np.asarray(data["energy"]).copy()
        signal = np.asarray(data["cube"][i, j]).copy()
        raw = signal.copy()
        channels = ["EELS intensity"]
    else:
        with h5py.File(data["h5_path"], "r") as f:
            raw = np.asarray(f[f"ForceMap/0/{i}:{j}"][:]).copy()
        axis = raw[7].copy()
        signal = raw[2].copy()
        channels = [f"ForceMap channel {k}" for k in range(raw.shape[0])]
    return {"axis": axis, "signal": signal, "raw": raw, "channels": channels}
