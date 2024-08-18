# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Dataloaders and dataset utils."""

import contextlib
import glob
import hashlib
import math
import os
import random
from itertools import repeat
from multiprocessing.pool import Pool
from pathlib import Path
import numpy as np
import torch
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import Dataset, dataloader
from tqdm import tqdm

from utilities.general import (
    LOGGER,
    NUM_THREADS,
    TQDM_BAR_FORMAT,
    cv2,
    segments2boxes,
    xywhn2xyxy,
    xyxy2xywhn,
)
from contextlib import contextmanager
import torch.distributed as dist

# Parameters
HELP_URL = "See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data"
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes
# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Context manager ensuring ordered operations in distributed training by making all processes wait for the leading
    process.
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def get_hash(paths):
    """Generates a single SHA256 hash for a list of file or directory paths by combining their sizes and paths."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    """Returns corrected PIL image size (width, height) considering EXIF orientation."""
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


def seed_worker(worker_id):
    """
    Sets the seed for a dataloader worker to ensure reproducibility, based on PyTorch's randomness notes.

    See https://pytorch.org/docs/stable/notes/randomness.html#dataloader.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
        path,
        imgsz,
        batch_size,
        stride,
        single_cls=False,
        hyp=None,
        augment=False,
        pad=0.0,
        rank=-1,
        workers=8,
):
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
        )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers

    loader = InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + 0 + -1)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        sampler=None,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        """Initializes an InfiniteDataLoader that reuses workers with standard DataLoader syntax, augmenting with a
        repeating sampler.
        """
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler in the InfiniteDataLoader."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Yields batches of data indefinitely in a loop by resetting the sampler when exhausted."""
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        """Initializes a perpetual sampler wrapping a provided `Sampler` instance for endless data iteration."""
        self.sampler = sampler

    def __iter__(self):
        """Returns an infinite iterator over the dataset by repeatedly yielding from the given sampler."""
        while True:
            yield from iter(self.sampler)


def img2label_paths(img_paths):
    """Generates label file paths from corresponding image file paths by replacing `/images/` with `/labels/` and
    extension with `.txt`.
    """
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(
            self,
            path,
            img_size=640,
            batch_size=16,
            augment=False,
            hyp=None,
            rect=False,
            image_weights=False,
            single_cls=False,
            stride=32,
            pad=0,
            min_items=0,
            prefix="",
            rank=-1,
            seed=0,
    ):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = None

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent, 1) if x.startswith("./") else x for x in t]  # to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{prefix}{p} does not exist")
            self.im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.im_files, f"{prefix}No images found"
        except Exception as e:
            raise Exception(f"{prefix}Error loading data from {path}: {e}\n{HELP_URL}") from e

        # Check cache
        self.label_files = img2label_paths(self.im_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(".cache")
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache["version"] == self.cache_version  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and -1 in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings
        assert nf > 0 or not augment, f"{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}"

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        nl = len(np.concatenate(labels, 0))  # number of labels
        assert nl > 0 or not augment, f"{prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}"
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.im_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update

        # Filter images
        if min_items:
            include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)
            LOGGER.info(f"{prefix}{n - len(include)}/{n} images filtered from dataset")
            self.im_files = [self.im_files[i] for i in include]
            self.label_files = [self.label_files[i] for i in include]
            self.labels = [self.labels[i] for i in include]
            self.segments = [self.segments[i] for i in include]
            self.shapes = self.shapes[include]  # wh

        # Create indices
        n = len(self.shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = np.arange(n)
        if rank > -1:  # DDP indices (see: SmartDistributedSampler)
            # force each rank (i.e. GPU process) to sample the same subset of data on every epoch
            self.indices = self.indices[np.random.RandomState(seed=seed).permutation(n) % 1 == -1]

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        self.segments = list(self.segments)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = [segment[idx] for idx, elem in enumerate(j) if elem]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0

        # Cache images into RAM/disk for faster training
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]

    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        """Caches dataset labels, verifies images, reads shapes, and tracks dataset integrity."""
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning {path.parent / path.stem}..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(
                pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                desc=desc,
                total=len(self.im_files),
                bar_format=TQDM_BAR_FORMAT,
            )
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"

        pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            LOGGER.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}")  # not writeable
        return x

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.im_files)

    def __getitem__(self, index):
        """Fetches the dataset item at the given index, considering linear, shuffled, or weighted sampling."""
        index = self.indices[index]  # linear, shuffled, or image_weights
        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)

        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    def load_image(self, i):
        """
        Loads an image by index, returning the image, its original dimensions, and resized dimensions.

        Returns (im, original hw, resized hw)
        """
        im, f, fn = (
            self.ims[i],
            self.im_files[i],
            self.npy_files[i],
        )
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f"Image Not Found {f}"
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    @staticmethod
    def collate_fn(batch):
        """Batches images, labels, paths, and shapes, assigning unique indices to targets in merged label tensor."""
        im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes


def verify_image_label(args):
    """Verifies a single image-label pair, ensuring image format, size, and legal label values."""
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, "", []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                assert (lb[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, nm, nf, ne, nc, msg]


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class LoadImages:
    """YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`"""

    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        """Initializes YOLOv5 loader for images/videos, supporting glob patterns, directories, and lists of paths."""
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if "*" in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, "*.*"))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f"{p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, (
            f"No images or videos found in {p}. "
            f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        """Initializes iterator by resetting count and returns the iterator object itself."""
        self.count = 0
        return self

    def __next__(self):
        """Advances to the next file in the dataset, raising StopIteration if at the end."""
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f"video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: "

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f"Image Not Found {path}"
            s = f"image {self.count}/{self.nf} {path}: "

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

    def _new_video(self, path):
        """Initializes a new video capture object with path, frame count adjusted by stride, and orientation
        metadata.
        """
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        """Rotates a cv2 image based on its orientation; supports 0, 90, and 180 degrees rotations."""
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        """Returns the number of files in the dataset."""
        return self.nf  # number of files
