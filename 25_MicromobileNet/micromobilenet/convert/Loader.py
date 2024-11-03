from typing import Callable, Tuple
from os.path import sep
from jinja2 import FileSystemLoader as Base


class Loader(Base):
    """
    Override default FileSystemLoader
    """
    def get_source(self, environment: "Environment", template: str) -> Tuple[str, str, Callable[[], bool]]:
        """

        :param environment:
        :param template:
        :return:
        """
        # normalize path separator for Windows and Unix
        template = template.replace(sep, "/")

        if not template.endswith(".jinja"):
            template = f"{template}.jinja"

        return super().get_source(environment, template)