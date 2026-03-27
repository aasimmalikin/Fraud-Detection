# Makes tests/ a Python package so imports resolve correctly across test files.
# Without this file, pytest may fail to resolve imports like:
#   from src.preprocessing import NullImputer
# when running from certain working directories.
