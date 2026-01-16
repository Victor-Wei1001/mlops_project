import os

_TEST_ROOT = os.path.dirname(__file__)  # 指向 tests 文件夹
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # 指向 mlops_project 根目录
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # 指向 data 文件夹
_PATH_SRC = os.path.join(_PROJECT_ROOT, "src")  # 指向 src 文件夹