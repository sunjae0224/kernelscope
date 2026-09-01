from kernelscope.workload import Workload


class PluginRegistry:
    def __init__(self):
        self._classes: dict[str, type] = {}

    def register(self, cls):
        if cls.name in self._classes:
            raise ValueError(f"plugin {cls.name!r} already registered")
        self._classes[cls.name] = cls
        return cls

    def names(self) -> list[str]:
        return list(self._classes)

    def get(self, name: str, **kwargs):
        try:
            cls = self._classes[name]
        except KeyError:
            raise KeyError(
                f"unknown plugin {name!r}; available: {', '.join(self._classes) or '(none)'}"
            ) from None
        return cls(**kwargs)

    def supporting(self, w: Workload) -> list:
        return [p for p in (self.get(n) for n in self._classes) if p.supports(w)]
