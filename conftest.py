# Root conftest.py — establishes the pytest rootdir and enables running
# all app test suites together with a single `pytest` invocation from the
# repo root without module-name collisions.
#
# Each app's tests/ directory has no __init__.py, so pytest uses the
# filesystem path as the unique module identifier (rootdir-relative import).
