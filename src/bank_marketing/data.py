"""Validate raw data, split first, fit preprocessing on train only."""
from dataclasses import dataclass
import hashlib
import io
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODES = ('realistic_pre_contact', 'duration_inclusive')
NUMERIC = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATEGORICAL = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
FEATURES = NUMERIC + CATEGORICAL


def validate_frame(frame):
    frame = frame.copy()
    missing = set(FEATURES + ['y']) - set(frame.columns)
    extra = set(frame.columns) - set(FEATURES + ['y'])
    if missing or extra:
        raise ValueError(f'Schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}')
    nan_counts = frame.isna().sum().to_dict()
    if any(nan_counts.values()):
        raise ValueError(f'Actual NaN policy: reject dataset; counts={nan_counts}')
    if set(frame.y.unique()) != {'yes', 'no'}:
        raise ValueError('Target must contain both yes and no only')
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        if not np.isfinite(frame[col]).all():
            raise ValueError(f'Non-finite numeric feature: {col}')
    for col in CATEGORICAL:
        frame[col] = frame[col].astype(str)
    y = frame.y.map({'no': 0, 'yes': 1}).to_numpy(dtype=np.int64)
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()
    metadata = {'name': 'UCI Bank Marketing', 'uci_id': 222, 'total_rows': len(frame),
                'positive': int(y.sum()), 'negative': int((y == 0).sum()),
                'actual_nan_counts': nan_counts, 'nan_policy': 'reject actual NaN; retain literal unknown',
                'unknown_counts': {c: int((frame[c] == 'unknown').sum()) for c in CATEGORICAL},
                'validated_frame_sha256': digest}
    return frame[FEATURES], y, metadata


def load_dataset(csv_path=None):
    if csv_path:
        frame = pd.read_csv(csv_path, sep=None, engine='python', keep_default_na=False, na_values=[''])
    else:
        # Original archive preserves literal unknown; standardized UCI API CSV has blanks.
        url = 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'
        with urllib.request.urlopen(url, timeout=120) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer:
            with zipfile.ZipFile(io.BytesIO(outer.read('bank.zip'))) as inner:
                raw = inner.read('bank-full.csv')
        frame = pd.read_csv(io.BytesIO(raw), sep=';', keep_default_na=False, na_values=[''])
    x, y, metadata = validate_frame(frame)
    metadata['source'] = 'local original CSV' if csv_path else url
    metadata['variant'] = 'bank-full.csv (45211 rows, 16 original features)'
    metadata['day_semantics'] = 'last contact day of month in bank-full.csv'
    if not csv_path:
        metadata['archive_sha256'] = hashlib.sha256(archive_bytes).hexdigest()
        metadata['raw_csv_sha256'] = hashlib.sha256(raw).hexdigest()
    return x, y, metadata



def split_indices(y, seed=42):
    train_val, test = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
    train, val = train_test_split(train_val, test_size=.2, stratify=y[train_val], random_state=seed)
    return {'train': train, 'validation': val, 'test': test}


def build_preprocessor(train_x, feature_mode=MODES[0]):
    if feature_mode not in MODES:
        raise ValueError(f'Unknown feature mode: {feature_mode}')
    numeric = [c for c in NUMERIC if c != 'duration' or feature_mode == MODES[1]]
    return ColumnTransformer([
        ('numeric', StandardScaler(), numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
    ], remainder='drop').fit(train_x)


@dataclass
class ArrayData:
    x: np.ndarray
    y: np.ndarray


def transform_split(preprocessor, x, y, indices):
    values = preprocessor.transform(x.iloc[indices]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite transformed data')
    return ArrayData(values, y[indices].copy())


def synthetic_frame(n=200, seed=7):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC})
    for col in CATEGORICAL:
        frame[col] = rng.choice(['a', 'b', 'unknown'], n)
    frame['y'] = np.where(np.arange(n) % 5 == 0, 'yes', 'no')
    return frame

