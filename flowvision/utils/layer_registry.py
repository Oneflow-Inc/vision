from flowvision.utils.base_registry import Registry
from typing import Any
import inspect
from tabulate import tabulate


class Layer_Registry(Registry):
    def __init__(self, name):
        super(Layer_Registry, self).__init__(name)
        self._obj_args = {}

    def _do_register(self, name: str, obj: Any) -> None:
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj
        if inspect.isfunction(obj):
            self._obj_args[name] = inspect.getfullargspec(obj)[
                0
            ]  # get the args of a function
        elif inspect.isclass(obj):
            self._obj_args[name] = inspect.getfullargspec(obj)[0][
                1:
            ]  # ignore the 'self' args in class

    def get_zip_dict(self) -> list:
        registered_obj_name = self._obj_map.keys()
        registered_obj_object = self._obj_map.values()
        registered_obj_args = self._obj_args.values()
        return zip(registered_obj_name, registered_obj_object, registered_obj_args)

    def __str__(self) -> str:
        table_headers = ["Names", "Objects", "Args"]
        table_items = self.get_zip_dict()
        table = tabulate(table_items, headers=table_headers, tablefmt="fancy_grid")
        return "Registry of {}:\n".format(self._name) + table

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects", "Args"]
        table_items = self.get_zip_dict()
        table = tabulate(table_items, headers=table_headers, tablefmt="fancy_grid")
        return "Registry of {}:\n".format(self._name) + table
