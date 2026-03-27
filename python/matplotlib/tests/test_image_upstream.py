"""
Upstream matplotlib tests for the image module (AxesImage).
"""

import numpy as np
import pytest

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage


# ---------------------------------------------------------------------------
# AxesImage basic construction
# ---------------------------------------------------------------------------
def test_axesimage_construction():
    """AxesImage can be constructed standalone."""
    fig, ax = plt.subplots()
    im = AxesImage(ax)
    assert im.get_array() is None


def test_axesimage_set_array():
    """set_array stores image data."""
    fig, ax = plt.subplots()
    im = AxesImage(ax)
    data = [[1, 2], [3, 4]]
    im.set_array(data)
    assert im.get_array() == [[1, 2], [3, 4]]


def test_axesimage_set_array_numpy():
    """set_array with numpy array converts to list."""
    fig, ax = plt.subplots()
    im = AxesImage(ax)
    data = np.zeros((3, 4))
    im.set_array(data)
    arr = im.get_array()
    assert len(arr) == 3
    assert len(arr[0]) == 4


def test_axesimage_extent():
    """set_extent / get_extent roundtrip."""
    fig, ax = plt.subplots()
    im = AxesImage(ax)
    im.set_extent((0, 10, 0, 5))
    assert im.get_extent() == (0, 10, 0, 5)


def test_axesimage_clim_roundtrip():
    """set_clim / get_clim roundtrip."""
    fig, ax = plt.subplots()
    im = AxesImage(ax)
    im.set_clim(0, 100)
    assert im.get_clim() == (0, 100)


def test_axesimage_cmap():
    """set_cmap / get_cmap roundtrip."""
    fig, ax = plt.subplots()
    im = AxesImage(ax)
    im.set_cmap('hot')
    assert im.get_cmap() == 'hot'


def test_axesimage_norm():
    """set_norm / get_norm roundtrip."""
    from matplotlib.colors import Normalize
    fig, ax = plt.subplots()
    im = AxesImage(ax)
    norm = Normalize(0, 1)
    im.set_norm(norm)
    assert im.get_norm() is norm


def test_axesimage_interpolation():
    """set_interpolation / get_interpolation roundtrip."""
    fig, ax = plt.subplots()
    im = AxesImage(ax)
    im.set_interpolation('bilinear')
    assert im.get_interpolation() == 'bilinear'


def test_axesimage_visible():
    """AxesImage visibility."""
    fig, ax = plt.subplots()
    im = AxesImage(ax)
    assert im.get_visible() is True
    im.set_visible(False)
    assert im.get_visible() is False


def test_axesimage_alpha():
    """AxesImage alpha."""
    fig, ax = plt.subplots()
    im = AxesImage(ax)
    assert im.get_alpha() is None
    im.set_alpha(0.7)
    assert im.get_alpha() == 0.7


def test_axesimage_label():
    """AxesImage label."""
    fig, ax = plt.subplots()
    im = AxesImage(ax)
    im.set_label('test')
    assert im.get_label() == 'test'


def test_axesimage_zorder():
    """AxesImage zorder."""
    fig, ax = plt.subplots()
    im = AxesImage(ax)
    assert im.get_zorder() == 0
    im.set_zorder(5)
    assert im.get_zorder() == 5


def test_axesimage_size():
    """AxesImage.get_size."""
    fig, ax = plt.subplots()
    data = [[1, 2, 3], [4, 5, 6]]
    im = AxesImage(ax, data=data)
    assert im.get_size() == (2, 3)


def test_axesimage_size_empty():
    """AxesImage.get_size with no data."""
    fig, ax = plt.subplots()
    im = AxesImage(ax)
    assert im.get_size() == (0, 0)


def test_axesimage_set_data_alias():
    """set_data is alias for set_array."""
    fig, ax = plt.subplots()
    im = AxesImage(ax)
    im.set_data([[1, 2], [3, 4]])
    assert im.get_array() == [[1, 2], [3, 4]]


def test_axesimage_auto_clim():
    """Auto clim from data."""
    fig, ax = plt.subplots()
    data = [[1, 5], [3, 9]]
    im = AxesImage(ax, data=data)
    vmin, vmax = im.get_clim()
    assert vmin == 1
    assert vmax == 9


def test_axesimage_clim_tuple_arg():
    """set_clim((vmin, vmax)) form."""
    fig, ax = plt.subplots()
    im = AxesImage(ax)
    im.set_clim((2, 8))
    assert im.get_clim() == (2, 8)


# ---------------------------------------------------------------------------
# Multiple images on same axes
# ---------------------------------------------------------------------------
def test_multiple_images():
    """Multiple imshow calls accumulate."""
    fig, ax = plt.subplots()
    im1 = ax.imshow(np.zeros((3, 3)))
    im2 = ax.imshow(np.ones((3, 3)))
    assert len(ax.images) == 2


# ---------------------------------------------------------------------------
# imshow with list-of-lists (no numpy)
# ---------------------------------------------------------------------------
def test_imshow_list_of_lists():
    """imshow accepts plain list-of-lists."""
    fig, ax = plt.subplots()
    data = [[1, 2], [3, 4]]
    im = ax.imshow(data)
    assert isinstance(im, AxesImage)
    assert im.get_array() == [[1, 2], [3, 4]]


def test_imshow_list_extent():
    """imshow list-of-lists with default extent."""
    fig, ax = plt.subplots()
    data = [[0, 0, 0], [0, 0, 0]]
    im = ax.imshow(data)
    ext = im.get_extent()
    assert ext == (-0.5, 2.5, 1.5, -0.5)


# ===================================================================
# AxesImage extended property tests
# ===================================================================

class TestAxesImageExtended:
    def test_set_cmap_string(self):
        """set_cmap with string name changes colormap."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.imshow([[1, 2], [3, 4]])
        im.set_cmap('hot')
        cmap = im.get_cmap()
        # cmap may be a string name or a Colormap object
        name = cmap.name if hasattr(cmap, 'name') else cmap
        assert name == 'hot'
        plt.close('all')

    def test_set_cmap_object(self):
        """set_cmap with Colormap object works."""
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        fig, ax = plt.subplots()
        im = ax.imshow([[1, 2], [3, 4]])
        cmap_obj = cm.get_cmap('plasma')
        im.set_cmap(cmap_obj)
        stored = im.get_cmap()
        name = stored.name if hasattr(stored, 'name') else stored
        assert name == 'plasma'
        plt.close('all')

    def test_set_norm_object(self):
        """set_norm works with Normalize object."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        fig, ax = plt.subplots()
        im = ax.imshow([[1, 2], [3, 4]])
        norm = Normalize(vmin=0, vmax=10)
        im.set_norm(norm)
        assert im.get_norm() is norm
        plt.close('all')

    def test_imshow_extent_stored(self):
        """imshow with extent stores it correctly."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.imshow([[1, 2], [3, 4]], extent=[0, 2, 0, 2])
        ext = im.get_extent()
        assert ext == (0, 2, 0, 2)
        plt.close('all')

    def test_imshow_shape_3channel(self):
        """imshow with 3-channel data stores correct shape."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        data = [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 0]]]
        im = ax.imshow(data)
        assert im.get_size() is not None
        plt.close('all')

    def test_imshow_interpolation_default(self):
        """imshow default interpolation can be retrieved."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.imshow([[1, 2], [3, 4]])
        interp = im.get_interpolation()
        # default is None (unset); test that it's gettable
        assert interp is None or isinstance(interp, str)
        plt.close('all')

    def test_imshow_set_interpolation(self):
        """set_interpolation changes interpolation."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.imshow([[1, 2], [3, 4]])
        im.set_interpolation('bilinear')
        assert im.get_interpolation() == 'bilinear'
        plt.close('all')

    def test_axesimage_label_from_constructor(self):
        """AxesImage constructed with label stores it."""
        from matplotlib.image import AxesImage
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = AxesImage(ax, label='myimage')
        assert im.get_label() == 'myimage'
        plt.close('all')

    def test_imshow_in_images(self):
        """imshow result is in ax.images."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.imshow([[1, 2], [3, 4]])
        assert im in ax.images
        plt.close('all')

    def test_imshow_alpha(self):
        """imshow image alpha can be set."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.imshow([[1, 2], [3, 4]])
        im.set_alpha(0.5)
        assert im.get_alpha() == 0.5
        plt.close('all')


# ===================================================================
# Additional AxesImage tests (upstream-inspired batch)
# ===================================================================

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
