from MG_main import multiply


def test_multiply():
    """This is a function to test the multiplication function"""
    assert multiply(3, 5) == 15
    assert multiply(4, 6) == 24
    assert multiply(0, 2) == 0


if __name__ == "__main__":
    test_multiply()
