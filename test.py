
from streamlit.testing.v1 import AppTest

def test_project_main():
    at = AppTest.from_file("projec_main.py").run()
    assert not at.exception

    # at.text_input('text').input('text').run()
    # assert at.warning[0].value == 'Sorry, the text did not match'