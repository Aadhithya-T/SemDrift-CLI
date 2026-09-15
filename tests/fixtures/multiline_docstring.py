"""
Fixture: Multiline docstring with PEP 257 indentation.
"""


def format_address(street: str, city: str, zip_code: str) -> str:
    """Format an address string cleanly.

    Parameters:
        street: The street name and number.
        city: City or municipality name.
        zip_code: Postal code string.

    Returns:
        Formatted single-line mailing address string.
    """
    return f"{street}, {city} {zip_code}"
