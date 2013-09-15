nalgebra
========

**nalgebra** is a _n_-dimensional linear algebra library written with the rust
programming language.


It is mainly focused on features needed for real-time physics. It should be
usable for graphics too.

## Compilation
You will need the last rust compiler from the master branch.
If you encounter problems, make sure you have the last version before creating an issue.

    git clone git://github.com/sebcrozet/nalgebra.git
    cd nalgebra
    make

## Design note
**nalgebra** is mostly written with non-idiomatic rust code. This is mostly because of limitations
of the trait system not allowing (easy) multiple overloading. Those overloading problems ares
worked around by this
[hack](http://smallcultfollowing.com/babysteps/blog/2012/10/04/refining-traits-slash-impls/)
(section _What if I want overloading_).
