import sys
import pytest
def run_all_tests(test_path: str = "app/tests") -> int:
    print(f"Starting test runner for: {test_path}...\n" + "-" * 40)

    pytest_args = [
        "-v",
        "--tb=short",
        test_path
    ]

    exit_code = pytest.main(pytest_args)

    print("-" * 40)
    if exit_code == 0:
        print("SUCCESS: All tests passed!")
    elif exit_code == 1:
        print("FAILED: One or more tests failed.")
    elif exit_code == 5:
        print("WARNING: No tests were found in the specified path.")
    else:
        print(f"est suite exited with unknown code: {exit_code}")

    return int(exit_code)

if __name__ == "__main__":
    run_all_tests("app/tests/test_report_service.py")
    run_all_tests("app/tests/test_inference_service.py")
    sys.exit(run_all_tests("app/tests/test_dataset_repository.py"))

