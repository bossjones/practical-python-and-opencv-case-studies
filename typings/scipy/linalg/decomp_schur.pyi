"""
This type stub file was generated by pyright.
"""

"""Schur decomposition functions."""
_double_precision = ...
def schur(a, output=..., lwork=..., overwrite_a=..., sort=..., check_finite=...): # -> tuple[Unknown, Unknown] | tuple[Unknown, Unknown, Unknown]:
    """
    Compute Schur decomposition of a matrix.

    The Schur decomposition is::

        A = Z T Z^H

    where Z is unitary and T is either upper-triangular, or for real
    Schur decomposition (output='real'), quasi-upper triangular. In
    the quasi-triangular form, 2x2 blocks describing complex-valued
    eigenvalue pairs may extrude from the diagonal.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to decompose
    output : {'real', 'complex'}, optional
        Construct the real or complex Schur decomposition (for real matrices).
    lwork : int, optional
        Work array size. If None or -1, it is automatically computed.
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance).
    sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
        Specifies whether the upper eigenvalues should be sorted. A callable
        may be passed that, given a eigenvalue, returns a boolean denoting
        whether the eigenvalue should be sorted to the top-left (True).
        Alternatively, string parameters may be used::

            'lhp'   Left-hand plane (x.real < 0.0)
            'rhp'   Right-hand plane (x.real > 0.0)
            'iuc'   Inside the unit circle (x*x.conjugate() <= 1.0)
            'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

        Defaults to None (no sorting).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    T : (M, M) ndarray
        Schur form of A. It is real-valued for the real Schur decomposition.
    Z : (M, M) ndarray
        An unitary Schur transformation matrix for A.
        It is real-valued for the real Schur decomposition.
    sdim : int
        If and only if sorting was requested, a third return value will
        contain the number of eigenvalues satisfying the sort condition.

    Raises
    ------
    LinAlgError
        Error raised under three conditions:

        1. The algorithm failed due to a failure of the QR algorithm to
           compute all eigenvalues.
        2. If eigenvalue sorting was requested, the eigenvalues could not be
           reordered due to a failure to separate eigenvalues, usually because
           of poor conditioning.
        3. If eigenvalue sorting was requested, roundoff errors caused the
           leading eigenvalues to no longer satisfy the sorting condition.

    See also
    --------
    rsf2csf : Convert real Schur form to complex Schur form

    Examples
    --------
    >>> from scipy.linalg import schur, eigvals
    >>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
    >>> T, Z = schur(A)
    >>> T
    array([[ 2.65896708,  1.42440458, -1.92933439],
           [ 0.        , -0.32948354, -0.49063704],
           [ 0.        ,  1.31178921, -0.32948354]])
    >>> Z
    array([[0.72711591, -0.60156188, 0.33079564],
           [0.52839428, 0.79801892, 0.28976765],
           [0.43829436, 0.03590414, -0.89811411]])

    >>> T2, Z2 = schur(A, output='complex')
    >>> T2
    array([[ 2.65896708, -1.22839825+1.32378589j,  0.42590089+1.51937378j],
           [ 0.        , -0.32948354+0.80225456j, -0.59877807+0.56192146j],
           [ 0.        ,  0.                    , -0.32948354-0.80225456j]])
    >>> eigvals(T2)
    array([2.65896708, -0.32948354+0.80225456j, -0.32948354-0.80225456j])

    An arbitrary custom eig-sorting condition, having positive imaginary part,
    which is satisfied by only one eigenvalue

    >>> T3, Z3, sdim = schur(A, output='complex', sort=lambda x: x.imag > 0)
    >>> sdim
    1

    """
    ...

eps = ...
feps = ...
_array_kind = ...
_array_precision = ...
_array_type = ...
def rsf2csf(T, Z, check_finite=...): # -> tuple[Any | Unknown, Any | Unknown]:
    """
    Convert real Schur form to complex Schur form.

    Convert a quasi-diagonal real-valued Schur form to the upper-triangular
    complex-valued Schur form.

    Parameters
    ----------
    T : (M, M) array_like
        Real Schur form of the original array
    Z : (M, M) array_like
        Schur transformation matrix
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    T : (M, M) ndarray
        Complex Schur form of the original array
    Z : (M, M) ndarray
        Schur transformation matrix corresponding to the complex form

    See Also
    --------
    schur : Schur decomposition of an array

    Examples
    --------
    >>> from scipy.linalg import schur, rsf2csf
    >>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
    >>> T, Z = schur(A)
    >>> T
    array([[ 2.65896708,  1.42440458, -1.92933439],
           [ 0.        , -0.32948354, -0.49063704],
           [ 0.        ,  1.31178921, -0.32948354]])
    >>> Z
    array([[0.72711591, -0.60156188, 0.33079564],
           [0.52839428, 0.79801892, 0.28976765],
           [0.43829436, 0.03590414, -0.89811411]])
    >>> T2 , Z2 = rsf2csf(T, Z)
    >>> T2
    array([[2.65896708+0.j, -1.64592781+0.743164187j, -1.21516887+1.00660462j],
           [0.+0.j , -0.32948354+8.02254558e-01j, -0.82115218-2.77555756e-17j],
           [0.+0.j , 0.+0.j, -0.32948354-0.802254558j]])
    >>> Z2
    array([[0.72711591+0.j,  0.28220393-0.31385693j,  0.51319638-0.17258824j],
           [0.52839428+0.j,  0.24720268+0.41635578j, -0.68079517-0.15118243j],
           [0.43829436+0.j, -0.76618703+0.01873251j, -0.03063006+0.46857912j]])

    """
    ...

