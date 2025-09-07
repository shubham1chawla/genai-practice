import json
from typing import List


class Book:
    def __init__(self, **d):
        self.id = d["id"]
        self.name = d["name"]
        self.url = d["url"]

    @staticmethod
    def load(json_path: str) -> 'List[Book]':
        with open(json_path, "r") as file:
            return json.loads(file.read(), object_hook=lambda d: Book(**d))
