Examples
========

.. _examples:

.. note 

Opening a HIPO File
-------------------

To open a single HIPO file use the
``hipopy.hipopy.open`` function.

The ``mode`` parameter should be either ``"r"``, ``"w"``,
or ``"a"`` (read, write, and append).

For example:

>>> import hipopy.hipopy
>>> f = hipopy.hipopy.open('file.hipo',mode='r')
>>> f.show()
               NEW::bank :     1     1     3
>>> f.read('NEW::bank')