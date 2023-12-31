from __future__ import annotations

import warp_runtime_py as m


def test_version():
    assert m.__version__ == "0.0.1"


def test_add():
    assert m.add(1, 2) == 3


def test_sub():
    assert m.subtract(1, 2) == -1


if __name__ == '__main__':
    print(m.add(2, 3))
    m.init()
    print(m.cuda_driver_version())
    print(m.cuda_toolkit_version())
    print(m.cuda_device_get_name(0))
