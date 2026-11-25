"""safe_write：寫到一半失敗、取代失敗時原檔不變、不留暫存檔；被鎖住時重試一次。執行：python tests/test_atomic_save.py"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import safe_write  # noqa: E402

D = tempfile.mkdtemp()
P = os.path.join(D, 'config.json')


def leftovers():
    return [f for f in os.listdir(D) if f.endswith('.tmp')]


def main():
    safe_write.write_json_atomic(P, {'a': 1})
    assert json.load(open(P)) == {'a': 1} and not leftovers()
    before = open(P, 'rb').read()

    real_w, real_r = safe_write._write_text, os.replace

    def half(fd, text):
        os.write(fd, text[:3].encode())
        os.close(fd)
        raise OSError(5, 'I/O error')
    safe_write._write_text = half
    try:
        safe_write.write_json_atomic(P, {'a': 2})
        raise AssertionError
    except safe_write.SaveError as e:
        assert e.file == 'config.json' and isinstance(e.cause, OSError)
    safe_write._write_text = real_w
    assert open(P, 'rb').read() == before and not leftovers()
    print('ok 寫到一半失敗')

    n = [0]

    def locked(a, b):
        n[0] += 1
        if n[0] == 1:
            raise PermissionError(13, 'locked')
        return real_r(a, b)
    os.replace = locked
    safe_write.RETRY_WAIT_S = 0.01
    try:
        safe_write.write_json_atomic(P, {'a': 3})
    finally:
        os.replace = real_r
    assert n[0] == 2 and json.load(open(P)) == {'a': 3} and not leftovers()
    print('ok 被鎖住時重試一次')

    os.replace = lambda a, b: (_ for _ in ()).throw(PermissionError(13, 'locked'))
    try:
        safe_write.write_json_atomic(P, {'a': 4})
        raise AssertionError
    except safe_write.SaveError:
        pass
    finally:
        os.replace = real_r
    assert json.load(open(P)) == {'a': 3} and not leftovers()
    print('ok 一直被鎖住：原檔不變')

    try:
        safe_write.write_json_atomic(P, {'a': object()})
        raise AssertionError
    except safe_write.SaveError:
        pass
    assert json.load(open(P)) == {'a': 3}
    print('全部通過')


if __name__ == '__main__':
    main()
