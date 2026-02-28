Don't go too deep without exposing and sanity checking your approach with the user.

Try to avoid thinking deeply unless explicity instructed to do so.

Don't worry about linting errors if `ruff check --fix` and `ruff format` will fix them.

# Docstrings

Write docstrings for public functions/methods.

For functions, they should be on this format

```
"""<short description>

<optional long description>

Args:
    arg1: <description>
    arg2: <description>

Returns:
    <description>
    <description>
"""
```

Never say types in descriptions, as they are already annotated. Mention the shape of all numpy arrays.
Try not to duplicate information inside the docstring too much, and don't just repeat the variable names.

# Comments

Keep comments minimal, only add comments when the code needs explaining why it's doing something.
