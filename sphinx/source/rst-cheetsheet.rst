.. highlight:: none

RST cheetsheet
**************

Headings
********

::

  H1 -- Top of the Page Header
  ============================

H1 -- Top of the Page Header
============================

::

  H2 -- Page Sections
  ===================

H2 -- Page Sections
===================

::

  H3 -- Subsection
  ----------------

H3 -- Subsection
----------------

::

  H4 -- Subsubsection
  +++++++++++++++++++

H4 -- Subsubsection
+++++++++++++++++++

::

  A Subpoint
  ----------
  This is subpoint

A Subpoint
----------
This is subpoint

::

  A subsubpoint
  ++++++++++++++
  This is subsubpoint

A subsubpoint
++++++++++++++
This is subsubpoint


Decorations
***********

::

  horizontal line below

  ----

  horizontal line above

horizontal line below

----

horizontal line above

::

  | The first line of text.
  | The second line of text (new line).
  | ...

| The first line of text.
| The second line of text (new line).
| ...

::

  Vertical spacing
  Paragraph 1

  |

  Paragraph 2

Vertical spacing
Paragraph 1

|

Paragraph 2

Text style
**********

::

**stronger emphasis** and *emphasis*

**stronger emphasis** and *emphasis*

Math
****

::

  .. math::

      n_{\mathrm{offset}} = \sum_{k=0}^{N-1} s_k n_k

.. math::

    n_{\mathrm{offset}} = \sum_{k=0}^{N-1} s_k n_k


::

  .. _cg_links:

  Links
  *****

  This is the section we want to reference to.

  ...

  The following - :ref:`cg_links` - generates a link to the section with
  the defined label using this section heading as a link title.

  A link label and a reference can be defined in separate source files,
  but within one directory. Otherwise, use the external linking.


.. _cg_links:

Links
*****

This is the section we want to reference to.

...

The following - :ref:`cg_links` - generates a link to the section with
the defined label using this section heading as a link title.

A link label and a reference can be defined in separate source files,
but within one directory. Otherwise, use the external linking.


::

  External link: `Python <http://www.python.org/>`_

External link: `Python <http://www.python.org/>`_


Lists
*****

::

* bullet
* bullet point

* bullet
* bullet point

::

- Another bullet
- item2
- item3

- Another bullet
- item2
- item3

::

  #. enumerate
  #. Item 2.
  #. Item 3.

#. enumerate
#. Item 2.
#. Item 3.


::

  Definition
    explanation

  Pagebreaking
    Process of breaking pages

Definition
  explanation

Pagebreaking
  Process of breaking pages

information blocks
******************

::

  .. note::

    This is a note.

.. note::

  This is a note.

::

  .. warning::

    This is a warning.

.. warning::

  This is a warning.


::

  .. important::

    This is important.

.. important::

  This is important.

::

  .. caution::

    This is caution.

.. caution::

  This is caution.

::

  .. tip::

    This is tip.

.. tip::

  This is tip.

::

  .. seealso::

    This is seealso.

.. seealso::

  This is seealso.


tables
******

::

  .. table:: **Default flavors**

   ============  =========  ===============  =============
    Flavor         VCPUs      Disk (in GB)     RAM (in MB)
   ============  =========  ===============  =============
    m1.tiny        1          1                512
    m1.small       1          20               2048
    m1.medium      2          40               4096
    m1.large       4          80               8192
    m1.xlarge      8          160              16384
   ============  =========  ===============  =============

.. table:: **Default flavors**

 ============  =========  ===============  =============
  Flavor         VCPUs      Disk (in GB)     RAM (in MB)
 ============  =========  ===============  =============
  m1.tiny        1          1                512
  m1.small       1          20               2048
  m1.medium      2          40               4096
  m1.large       4          80               8192
  m1.xlarge      8          160              16384
 ============  =========  ===============  =============


::

  .. list-table:: **Quota descriptions**
     :widths: 10 25 10
     :header-rows: 1

     * - Quota Name
       - Defines the number of
       - Service
     * - Gigabytes
       - Volume gigabytes allowed for each project
       - Block Storage
     * - Instances
       - Instances allowed for each project.
       - Compute
     * - Injected File Content Bytes
       - Content bytes allowed for each injected file.
       - Compute


.. list-table:: **Quota descriptions**
   :widths: 10 25 10
   :header-rows: 1

   * - Quota Name
     - Defines the number of
     - Service
   * - Gigabytes
     - Volume gigabytes allowed for each project
     - Block Storage
   * - Instances
     - Instances allowed for each project.
     - Compute
   * - Injected File Content Bytes
     - Content bytes allowed for each injected file.
     - Compute


::

  .. csv-table:: **ipv6_ra_mode and ipv6_address_mode combinations**
     :header: ipv6 ra mode, ipv6 address mode, "radvd A,M,O", "External Router A,M,O", Description
     :widths: 2, 2, 2, 2, 4

     *N/S*, *N/S*, Off, Not Defined, Backwards compatibility with pre-Juno IPv6 behavior.
     *N/S*, slaac, Off, "1,0,0", Guest instance obtains IPv6 address from non-OpenStack
     *N/S*, dhcpv6-stateful, Off, "0,1,1", Not currently implemented in the reference implementation.

.. csv-table:: **ipv6_ra_mode and ipv6_address_mode combinations**
   :header: ipv6 ra mode, ipv6 address mode, "radvd A,M,O", "External Router A,M,O", Description
   :widths: 2, 2, 2, 2, 4

   *N/S*, *N/S*, Off, Not Defined, Backwards compatibility with pre-Juno IPv6 behavior.
   *N/S*, slaac, Off, "1,0,0", Guest instance obtains IPv6 address from non-OpenStack
   *N/S*, dhcpv6-stateful, Off, "0,1,1", Not currently implemented in the reference implementation.


code sample
***********

::

  .. code-block:: python

     def some_function():
         interesting = False
         print 'Hello World'


.. code-block:: python

   def some_function():
       interesting = False
       print 'Hello World'


::

  .. code-block:: python
     :linenos:
     :emphasize-lines: 3,5-6

     def some_function():
         interesting = False
         print 'This line is highlighted.'
         print 'This one is not...'
         print '...but this one is.'
         print 'This one is highlighted too.'


.. code-block:: python
   :linenos:
   :emphasize-lines: 3,5-6

   def some_function():
       interesting = False
       print 'This line is highlighted.'
       print 'This one is not...'
       print '...but this one is.'
       print 'This one is highlighted too.'


comment
*******

::

  .. This is a comment. It is not visible in the documentation build.
     Generally, use it to include TODO within the content followed
     by the initials of the person who is to perform the action.

.. This is a comment. It is not visible in the documentation build.
   Generally, use it to include TODO within the content followed
   by the initials of the person who is to perform the action.


.. versionadded:: version

.. versionchanged:: version


This is a statement.

.. versionadded:: 0.0.1

It's okay to use this code.


H1 -- Japanese/日本語
*********************
H2 -- えっちに
==============
H3 -- えっちさん
----------------
H4 -- えっちよん
++++++++++++++++
さぶぽいんと
------------
さぶさぶぽいんと
++++++++++++++++

漢字
ひらがな
カタカナ
１一壱

.. note::

  ノート

.. warning::

  警告


.. important::

  重要

.. caution::

  注意

.. tip::

  ティップ

名前
  意味

定義
  説明


.. table:: **通常表**

 ============  =========  ================  ================
  Flavor        演算装置  ディスク (in GB)  メモリー (in MB)
 ============  =========  ================  ================
  m1.tiny        1          1                512
  m1.small       1          20               2048
  m1.medium      2          40               4096
  m1.large       4          80               8192
  m1.xlarge      8          160              16384
 ============  =========  ================  ================

