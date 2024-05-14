import logging
from collections import namedtuple
from math import ceil
from typing import Iterator

import numpy as np
from numpy import ndarray
from numpy.random import default_rng
from PIL import Image, ImageFilter
from skimage import filters

logger = logging.getLogger("__name__")

CoordDescriptor = namedtuple("CoordDescriptor", ["coord", "prob"])

# doc PIL methods: https://pillow.readthedocs.io/en/stable/reference/Image.html#the-image-class
# doc PIL filters: https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#module-PIL.ImageFilter


class NoSamplablePoints(Exception):
    def __init__(self) -> None:
        super().__init__("There are no samplable points with probability > 0.")


class PatchOverflow(Exception):
    def __init__(self) -> None:
        super().__init__("Sampled patch overflows the image edges")


class WrongPatchSize(Exception):
    def __init__(self) -> None:
        super().__init__("Wrong size of sampled patch")


class WrongImpMapSize(Exception):
    def __init__(self) -> None:
        super().__init__("Wrong size of importance map")


class Patcher:
    """Class for sampling patches from an image

    Args:
        image_path (str): path to image
        patch_size (int): size of patch (only one integer, patch is square)
        reduce_factor (int): factor in case image has to be downscaled before patch sampling (set 1 for no downscaling)
        grid_sep (int): separation in pixels of grid points for sampling patches (half or third the size of the patch is a reasonable value to avoid too much overlap between sampled patches)
        map_type (str): 'importance' or 'uniform'
        patches_per_image (int): how many patches to sample per image
        scale_dog (float): scale of the difference of gaussians used to compute the importance map. Depends on the scale of the objects to be captured by the sampling. Try different values in notebook and look the results (reasonable values to try: 1, 2, 3, 4, 5)
        blur_samp_map (bool): whether to blur the importance map (qualitative results seem to be better WITHOUT blurring, but letting the option nevertheless)
        return_samp_map (bool): for debug purposes, return a section of the importance map along with the patch for each sample
        seed (int): asdf
    """

    def __init__(
        self,
        image_path: str,
        patch_size: int,
        reduce_factor: int,
        grid_sep: int,
        map_type: str,
        patches_per_image: int,
        scale_dog: float,
        blur_samp_map: bool = True,
        return_samp_map: bool = False,
        seed: int | None = None,
    ) -> None:
        self.path = image_path
        self.patch_size = patch_size
        self.reduce_factor = reduce_factor
        self.grid_sep = grid_sep
        self.map_type = map_type
        self.patches_per_image = patches_per_image
        self.scale_dog = scale_dog
        self.blur_samp_map = blur_samp_map
        self.return_samp_map = return_samp_map

        self.cur_patch = 0

        # read and post-process image
        self.image = load_image(image_path)
        self._postproc_image()

        # reduce image
        if self.reduce_factor > 1:
            self.image = self.image.reduce(self.reduce_factor)

        # compute and postprocess importance map
        self.imp_map = self._compute_sampling_map()
        self._postproc_importance_map()

        # compute coord dict for convenience of sampling with np.random.choice
        self.coord_dict = self._compute_coordinates_dictionary()

        # initialize random generator
        self.rng = default_rng(seed)

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> Image.Image | dict:
        """returns next patch until pre-defined amount is reached"""
        self.cur_patch += 1
        if self.cur_patch <= self.patches_per_image:
            # sample coordinate and get patch
            i, j = self._sample_imp_coords()
            ii, jj = self._imp_coords_to_image(i, j)
            patch = self.get_patch_at_coords(ii, jj)
            if self.return_samp_map:
                imp_patch = self.get_imp_patch_at_coords(i, j)
                return {"image": patch, "imp_patch": imp_patch, "imp_val": self.imp_map[i, j]}
            return patch
        raise StopIteration

    def _sample_imp_coords(self) -> tuple[int, int]:
        """samples coordinates according to importance map"""
        i, j = self.rng.choice(a=self.coord_dict["coord"], p=self.coord_dict["prob"])

        if self.imp_map[i, j] <= 0.0:
            raise NoSamplablePoints

        return i, j

    def get_patch_at_coords(self, h_coord: int, w_coord: int) -> Image.Image:
        """returns Pillow crop of the patch at given coordinates
        coordinates are in numpy order: 1st height, 2nd width
        """
        left = w_coord - (self.patch_size // 2)
        right = left + self.patch_size
        upper = h_coord - (self.patch_size // 2)
        lower = upper + self.patch_size

        # sanity check overflow
        im_w, im_h = self.image.size
        if left <= 0 or upper <= 0 or right >= im_w or lower >= im_h:
            raise PatchOverflow

        # extract patch
        patch = self.image.crop((left, upper, right, lower))

        # sanity check patch size
        p_w, p_h = patch.size
        if p_w != self.patch_size or p_h != self.patch_size:
            raise WrongPatchSize

        return patch

    def get_imp_patch_at_coords(self, h: int, w: int) -> Image.Image:
        """returns a crop of the importance map at given coordinates
        coordinates are in numpy order: 1st height, 2nd width
        """
        return self.imp_map[h - self.margin : h + self.margin, w - self.margin : w + self.margin]

    def _imp_coords_to_image(self, i: int, j: int) -> tuple[int, int]:
        """maps coordinates from importance map to original image space"""
        im_w, im_h = self.image.size
        map_h, map_w = self.imp_map.shape
        f_h = float(im_h / map_h)  # scaling factor
        f_w = float(im_w / map_w)
        new_i = round((i + 0.5) * f_h)  # add 0.5 to sample from center of grid cell
        new_j = round((j + 0.5) * f_w)
        return new_i, new_j

    def _postproc_importance_map(self) -> None:
        """post-process importance map"""

        # zeroes-out margin to avoid sampling patches that overflow the edges
        self.margin = int(ceil(self.patch_size / (2.0 * self.grid_sep)))  # 2 in denom bc half of patch
        self.margin = max(2, self.margin)  # minimum sanity margin of 2 to counteract BlurBox artifacts
        self.imp_map = _zero_edges(self.imp_map, self.margin, self.map_type)

    def _compute_coordinates_dictionary(self) -> dict:
        """creates a descriptor with coordinates along with their associated probabilities.
        For convenient use with np.random.choice
        """

        I, J = np.meshgrid(  # noqa: E741
            range(self.imp_map.shape[0]),
            range(self.imp_map.shape[1]),
            indexing="ij",
        )  # pylint: disable=invalid-name

        sum_total = self.imp_map.sum()
        if np.isclose(sum_total, 0.0):
            logger.warning(f"overall importance zero in importance map for image {self.path}")
            logger.warning("reverting to uniform sampling for this case")

            self.map_type = "uniform"
            self.imp_map[:] = 1.0
            self._postproc_importance_map()

            sum_total = self.imp_map.sum()

        probs = self.imp_map / sum_total  # probabilities

        coord_desc = [CoordDescriptor((i, j), probs[i, j]) for i, j in zip(I.ravel(), J.ravel())]

        return _descriptor_to_dict(coord_desc)

    def _postproc_image(self) -> None:
        """post-process image"""

        # converts to 8-bit grayscale (required for some operations, eg, reduce)
        # https://stackoverflow.com/questions/8062564/cant-apply-image-filters-on-16-bit-tifs-in-pil
        try:
            if self.image.mode == "I;16":
                # self.image.mode = "I"  # type: ignore[misc]
                self.image = self.image.point(lambda i: i * (1.0 / 256)).convert("L")

        except Exception as exc:
            logger.error("failed to post-process the image")
            raise exc

    def get_importance_map(self) -> ndarray:
        """returns importance map (for dataset inspection)"""
        return self.imp_map

    def get_image_thumbnail(self, size: int = 128) -> ndarray:
        """returns a thumbnail fo the image (for dataset inspection)"""
        thumb = self.image.copy()
        thumb.thumbnail((size, size))
        return np.array(thumb)

    def _compute_sampling_map(self) -> ndarray:
        """computes sampling map from image to sample the patches"""
        im_w, im_h = self.image.size
        map_w = int(ceil(float(im_w) / self.grid_sep))
        map_h = int(ceil(float(im_h) / self.grid_sep))

        imp_map_np = np.array([], dtype="float64")
        if self.map_type == "importance":
            imp_map = compute_thresholded_dog(self.image, self.scale_dog)
            if self.blur_samp_map:
                imp_map = imp_map.filter(ImageFilter.BoxBlur(self.grid_sep))
            imp_map = imp_map.reduce(self.grid_sep)
            imp_map_np = np.array(imp_map, dtype="float64")
        elif self.map_type == "uniform":
            imp_map_np = np.ones((map_h, map_w), dtype="float64")
        else:
            raise ValueError

        # assure map size before returning
        height, width = imp_map_np.shape
        if width != map_w or height != map_h:
            raise WrongImpMapSize

        return imp_map_np


def load_image(path: str) -> Image.Image:
    """loads image from path"""
    try:
        return Image.open(path)
    except Exception as exc:
        logger.error(f"failed to load image {path}")
        raise exc


def _zero_edges(arr: ndarray, mrg: int, map_type: str = "importance") -> ndarray:
    """zeroes-out the edges of a 2D array"""
    arr2 = arr[mrg:-mrg, mrg:-mrg]
    min_val = 0.0  # in case uniform map
    if map_type == "importance":
        min_val = arr.min()
    arr2 = np.pad(arr2, [(mrg, mrg), (mrg, mrg)], constant_values=min_val)
    arr2 -= min_val
    if arr2.shape != arr.shape:
        raise ValueError
    return arr2


def _descriptor_to_dict(desc: list[CoordDescriptor]) -> dict:
    """Returns a dictionary where fields are zipped separately"""
    if not desc:
        logger.error("cannot zip an empty descriptor")
        raise ValueError
    fields = desc[0]._fields
    return dict(zip(fields, zip(*desc)))


def compute_thresholded_dog(image: Image.Image, scale: float) -> ndarray:
    """computes difference-of-gaussians image at given scale and thresholds it"""

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Compute difference of Gaussians
    # Apply Gaussian filter for two different scales
    gaussian1 = filters.gaussian(image_array, sigma=scale)
    gaussian2 = filters.gaussian(image_array, sigma=scale * 1.6)  # Multiplying by 1.6 to get the next scale

    # Compute the difference of Gaussians
    dog_image = gaussian2 - gaussian1

    # Apply thresholding to emphasize edges
    thr = filters.threshold_otsu(dog_image)
    thresholded_image = dog_image > thr

    # return PIL image
    return Image.fromarray(thresholded_image).convert("L")
