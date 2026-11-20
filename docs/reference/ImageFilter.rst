.. py:module:: PIL.ImageFilter
.. py:currentmodule:: PIL.ImageFilter

:py:mod:`~PIL.ImageFilter` module
=================================

The :py:mod:`~PIL.ImageFilter` module contains definitions for a pre-defined set of
filters, which can be be used with the :py:meth:`Image.filter()
<PIL.Image.Image.filter>` method.

Example: Filter an image
------------------------

::

    from PIL import ImageFilter

    im1 = im.filter(ImageFilter.BLUR)

    im2 = im.filter(ImageFilter.MinFilter(3))
    im3 = im.filter(ImageFilter.MinFilter)  # same as MinFilter(3)

Using a 3D ``.cube`` LUT
------------------------

``ImageFilter.CubeLUT`` loads a 3D LUT exported in the Cube LUT
Specification 1.0 format. It accepts either a file path or an open text
stream and can be used directly with :py:meth:`Image.filter()
<PIL.Image.Image.filter>`.

::

    from PIL import Image, ImageFilter

    lut = ImageFilter.CubeLUT("night.cube")
    with Image.open("screenshot.png") as im:
        result = im.filter(lut)
        result.save("graded.png")

An open text stream can be used as well. ``CubeLUT`` reads from its current
position and does not close it::

    with open("night.cube") as fp:
        lut = ImageFilter.CubeLUT(fp)
        result = im.filter(lut)

``TITLE``, ``LUT_3D_SIZE``, ``DOMAIN_MIN``, ``DOMAIN_MAX``, comments and
scientific notation are supported. Data rows use the Cube LUT ordering where
red changes fastest and blue changes slowest. When no input domain is
declared, each channel defaults to the range 0 to 1. Values outside a
declared domain are clamped to its nearest endpoint.

Colors are interpolated with the tetrahedral method recommended by the Cube
LUT specification. The table may contain finite values below 0 or above 1;
interpolation is performed before the final result is clamped to the 8-bit
output range. ``RGB`` and ``RGBA`` images are supported, and the alpha
channel of ``RGBA`` images is preserved unchanged. No color space conversion
is applied.

Only 3D LUTs are supported. 1D LUTs and combined 1D/3D LUTs are rejected
with an explicit error instead of being ignored.

Filters
-------

Pillow provides the following set of predefined image enhancement filters:

* **BLUR**
* **CONTOUR**
* **DETAIL**
* **EDGE_ENHANCE**
* **EDGE_ENHANCE_MORE**
* **EMBOSS**
* **FIND_EDGES**
* **SHARPEN**
* **SMOOTH**
* **SMOOTH_MORE**

.. autoclass:: PIL.ImageFilter.Color3DLUT
    :members:

.. autoclass:: PIL.ImageFilter.CubeLUT
    :members:

.. autoclass:: PIL.ImageFilter.BoxBlur
    :members:

.. autoclass:: PIL.ImageFilter.GaussianBlur
    :members:

.. autoclass:: PIL.ImageFilter.UnsharpMask
    :members:

.. autoclass:: PIL.ImageFilter.Kernel
    :members:

.. autoclass:: PIL.ImageFilter.RankFilter
    :members:

.. autoclass:: PIL.ImageFilter.MedianFilter
    :members:

.. autoclass:: PIL.ImageFilter.MinFilter
    :members:

.. autoclass:: PIL.ImageFilter.MaxFilter
    :members:

.. autoclass:: PIL.ImageFilter.ModeFilter
    :members:

.. class:: Filter

    An abstract mixin used for filtering images
    (for use with :py:meth:`~PIL.Image.Image.filter`).

    Implementors must provide the following method:

    .. method:: filter(self, image)

        Applies a filter to a single-band image, or a single band of an image.

        :returns: A filtered copy of the image.

.. class:: MultibandFilter

    An abstract mixin used for filtering multi-band images
    (for use with :py:meth:`~PIL.Image.Image.filter`).

    Implementors must provide the following method:

    .. method:: filter(self, image)

        Applies a filter to a multi-band image.

        :returns: A filtered copy of the image.
