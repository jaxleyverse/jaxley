from abc import ABC, abstractmethod


class Module(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self, u, dt, *args):
        raise NotImplementedError


class View:
    def __init__(self, pointer, view):
        self.pointer = pointer
        self.view = view

    def set_params(self, key, val):
        self.pointer.params[key][self.view.index.values] = val

    def adjust_view(self, key, index):
        self.view = self.view[self.view[key] == index]
        self.view -= self.view.iloc[0]
        return self
