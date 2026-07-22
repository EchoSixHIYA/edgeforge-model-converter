from abc import ABC, abstractmethod


class ConversionBackend(ABC):
    name = "base"
    output_suffix = ""

    @abstractmethod
    def convert(self, source, output, options, progress):
        """Convert source to output and report progress(percent, message)."""


class BackendRegistry:
    def __init__(self):
        self._backends = {}

    def register(self, backend):
        self._backends[backend.name] = backend

    def get(self, name):
        if name not in self._backends:
            raise ValueError(f"暂不支持输出格式: {name}")
        return self._backends[name]

    def describe(self):
        return [{"id": k, "suffix": v.output_suffix} for k, v in self._backends.items()]

