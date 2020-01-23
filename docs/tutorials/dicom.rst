Working with DICOM files
========================

All the utils for owrking with DICOM files are stored in
``dpipe.medim.dicom``:

.. code-block:: python3

    from dpipe.medim.dicom import *

Before we start analysing our files, let’s install some additional
libraries, which add support for various medical imaging formats:

.. code:: bash

   conda install -c glueviz gdcm # Python 3.5 and 3.6
   conda install -c conda-forge gdcm # Python 3.7

We’ll be working with a subset of the ``CT Lymph Nodes`` dataset which
can be downloaded
`here <https://wiki.cancerimagingarchive.net/display/Public/CT+Lymph+Nodes>`__.

.. code-block:: python3

    path = '~/dicom_data/'

Crawling
--------

``join_dicom_tree`` is the main function that collects the DICOM files
metadata:

.. code-block:: python3

    df = join_dicom_tree(path, relative=True, verbose=False) 

It recursively visits the subfolders of ``path``, also it adds some
additional attributes: ``NoError``, ``HasPixelArray``, ``PathToFolder``,
``FileName``:

.. code-block:: python3

    len(df), df.NoError.sum(), df.HasPixelArray.sum()




.. parsed-literal::

    (2588, 2587, 2587)



Thre resulting dataframe has 2588 files’ metadata in it, and only one
file was openned with errors, let’s check which one:

.. code-block:: python3

    df.loc[~df.NoError, ['FileName', 'PathToFolder']]




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>FileName</th>
          <th>PathToFolder</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>readme.txt</td>
          <td>.</td>
        </tr>
      </tbody>
    </table>
    </div>



There is a file ``readme.txt`` in the root of the folders tree, which is
obvisously not a DICOM file.

Note that ``PathToFolder`` is relative to ``path``, this is because we
passed ``relative=True`` to ``join_dicom_tree``.

.. code-block:: python3

    # leave only dicoms that contain images (Pixel Arrays)
    dicoms = df[df.NoError & df.HasPixelArray]
    
    dicoms.FileName[1], dicoms.PathToFolder[1]




.. parsed-literal::

    ('000466.dcm',
     'ABD_LYMPH_001/09-14-2014-ABDLYMPH001-abdominallymphnodes-30274/abdominallymphnodes-26828')



Aggregation
-----------

Next, we can join the dicom files into series, which are often easier to
operate with:

.. code-block:: python3

    images = aggregate_images(dicoms)
    len(images)




.. parsed-literal::

    4



``aggregate_images`` also adds some attributes: ``SlicesCount``,
``FileNames``, ``InstanceNumbers``, check its docstring for more
information.

For example ``FileNames`` contains all the files that are part of a
particular series:

.. code-block:: python3

    images.FileNames[0][:50] + '...'




.. parsed-literal::

    '000466.dcm/000312.dcm/000150.dcm/000357.dcm/000311...'



As you can see, they are not ordered by default, but you can change this
behaviour by passing the ``process_series`` argument which receives a
subset of the dataframe, containing files from the same series:

.. code-block:: python3

    images = aggregate_images(dicoms, process_series=lambda series: series.sort_values('FileName'))
    
    images.FileNames[0][:50] + '...'




.. parsed-literal::

    '000000.dcm/000001.dcm/000002.dcm/000003.dcm/000004...'



Loading
-------

You can load a particular series’ images stacked into a numpy array
using the following function:

.. code-block:: python3

    img = load_series(images.loc[0], path)

it expects a row from the aggregated dataframe and, optinally, the
``path`` argument, if the paths are relative.

The image’s orientation as well as the slices’ order are determined
automatically:

.. code-block:: python3

    print(img.shape, images.PixelArrayShape[0], images.SlicesCount[0])


.. parsed-literal::

    (512, 512, 661) 512,512 661


Finally, you can visualize the series using ``slice3d``:

.. code-block:: python3

    from dpipe.medim.visualize import slice3d
    
    slice3d(img)
