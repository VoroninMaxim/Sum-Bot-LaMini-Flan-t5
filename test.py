import pytest
from streamlit.testing.v1 import AppTest
from add import file_preprocessing

def test_project_main():
    at = AppTest.from_file("projec_main.py").run()
    assert not at.exception

def test_file_preprocessing():
    with open("test.pdf", "rb") as f:
        file = f.read()
    final_texts, _ = file_preprocessing(file)
    assert isinstance(final_texts, str)

if __name__ == "__main__":
    pytest.main([__file__])

    # at.text_input('text').input('text').run()
    # assert at.warning[0].value == 'Sorry, the text did not match'