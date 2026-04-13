E2E test harness

Place images for registration in `tests/register/` with filenames like `Alice_1.jpg`, `Bob_1.jpg` (the name before underscore will be used as the person name).

Place images to test recognition in `tests/recognize/`.

Activate the Python 3.11 venv created earlier and run:

    source .venv311/bin/activate
    python tests/e2e_test.py

The script will store encodings in the project's `attendance.db` and print recognition results.
