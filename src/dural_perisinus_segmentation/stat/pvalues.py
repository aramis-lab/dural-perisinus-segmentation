def pval_to_stars(p: float) -> str:
    """
    Determines the level of significance.

    Parameters
    ----------
    p : float
        A p-value.

    Returns
    -------
    str
        The number of stars.
    """
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return "ns"