import os
import io
import pickle

import lmdb
import torch.utils.data as data
from PIL import Image


def num_samples(dataset, train):
    if dataset == "celeba":
        return 27000 if train else 3000
    elif dataset == "celeba64":
        return 162770 if train else 19867
    elif dataset == "imagenet-oord":
        return 1281147 if train else 50000
    elif dataset == "ffhq":
        return 63000 if train else 7000
    else:
        raise NotImplementedError("dataset %s is unknown" % dataset)


class LMDBDataset(data.Dataset):
    def __init__(self, root, name="", split="val", transform=None, is_encoded=False):
        self.name = name
        self.transform = transform
        self.split = split

        if self.split == "train":
            lmdb_path = os.path.join(root, "train.lmdb")
        elif self.split == "val":
            lmdb_path = os.path.join(root, "valid.lmdb")
            if not os.path.isfile(lmdb_path):
                lmdb_path = os.path.join(root, "val.lmdb")
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.lmdb_path = lmdb_path
        self._lmdb_open_kwargs = {
            "readonly": True,
            "subdir": os.path.isdir(lmdb_path),
            "max_readers": 256,
            "lock": False,      # only safe if DB is immutable while reading
            "readahead": False,
            "meminit": False,
        }

        self.env = None
        self.txn = None
        self.is_encoded = is_encoded
        #NOTE: fix error of lmdb already open
        with lmdb.open(self.lmdb_path, **self._lmdb_open_kwargs) as env:
            with env.begin(write=False) as txn:
                self.length = pickle.loads(txn.get(b"__len__"))
                self.keys = pickle.loads(txn.get(b"__keys__"))
        self.env = None
        self.txn = None

    def open_lmdb(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, **self._lmdb_open_kwargs)
            self.txn = self.env.begin(write=False, buffers=True)

    def __getitem__(self, index):
        if self.txn is None:
            self.open_lmdb()

        byteflow = self.txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)

        imgbuf = unpacked[0]
        buf = io.BytesIO()
        buf.write(imgbuf[0])
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        return img, target, {"index": str(index).zfill(5) + ".png"}

    def __len__(self):
        return self.length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None
        state["txn"] = None
        return state


# class LMDBDataset(data.Dataset):
#     def __init__(self, root, name='', split='train', transform=None, is_encoded=False):
#         path = os.path.join(root, 'validation.lmdb')
#         self.env = lmdb.open(
#             path,
#             max_readers=32,
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False,
#         )

#         if not self.env:
#             raise IOError('Cannot open lmdb dataset', path)

#         with self.env.begin(write=False) as txn:
#             self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

#         self.resolution = 256
#         self.transform = transform
#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         with self.env.begin(write=False) as txn:
#             key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
#             img_bytes = txn.get(key)

#         buffer = io.BytesIO(img_bytes)
#         img = Image.open(buffer)
#         img = img.convert('RGB')
#         target = 0

#         return img, target, index
