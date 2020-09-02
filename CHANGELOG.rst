=============
Release notes
=============


Development version
===================

* The (quite basic and slow) Slater-Koster table generation
  functionality has been removed as this is not the focus
  of Tango. Specialized codes should now be used to generate
  the required ``*-*_no_repulsion.skf`` files. These include
  Hotbit_, the ONECENT and TWOCENT codes from the DFTB+
  developers, and Hotcent_.

.. _Hotbit: https://github.com/pekkosk/hotbit
.. _Hotcent: https://gitlab.com/mvdb/hotcent

* No more backwards compatibility with Python2 (only Python3).

* The `relax_alternate` function in tango.relax_utils has been
  renamed to `relax_standard`.

Version 0.9
===========

12 December 2019

* Start of versioning.
