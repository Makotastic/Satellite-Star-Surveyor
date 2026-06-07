from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar, Generic, Self, TypeVar, overload

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
FieldSpec = int | type["ArrayView"]
TArrayView = TypeVar("TArrayView", bound="ArrayView")


class ArrayView:
    __slots__ = ("data",)

    __fields__: ClassVar[list[tuple[str, FieldSpec]]] = []
    __aliases__: ClassVar[dict[str, str]] = {}
    __defaults__: ClassVar[dict[str, Any]] = {}

    SIZE: ClassVar[int]
    SLICES: ClassVar[dict[str, slice]]
    FIELD_SPECS: ClassVar[dict[str, FieldSpec]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        offset = 0
        slices: dict[str, slice] = {}
        specs: dict[str, FieldSpec] = {}

        for name, spec in cls.__fields__:
            if isinstance(spec, int):
                size = spec
            else:
                size = spec.SIZE

            sl = slice(offset, offset + size)
            slices[name] = sl
            specs[name] = spec

            setattr(cls, name, property(cls._make_field_getter(sl, spec)))

            offset += size

        cls.SIZE = offset
        cls.SLICES = slices
        cls.FIELD_SPECS = specs

        for alias_name, path in cls.__aliases__.items():
            cls._validate_alias_path(alias_name, path)
            setattr(cls, alias_name, property(cls._make_alias_getter(path)))

    def __init__(self, data: FloatArray):
        if data.shape != (self.SIZE,):
            raise ValueError(
                f"{type(self).__name__} expected shape {(self.SIZE,)}, "
                f"got {data.shape}"
            )

        self.data = data

    @staticmethod
    def _make_field_getter(sl: slice, spec: FieldSpec):
        def getter(self: ArrayView) -> FloatArray | ArrayView:
            view = self.data[sl]

            if isinstance(spec, int):
                return view

            return spec(view)

        return getter

    @staticmethod
    def _make_alias_getter(path: str):
        parts = path.split(".")

        def getter(self: ArrayView) -> Any:
            obj: Any = self
            for part in parts:
                obj = getattr(obj, part)
            return obj

        return getter

    @classmethod
    def _validate_alias_path(cls, alias_name: str, path: str) -> None:
        try:
            cls._slice_of_parts(path.split("."), base_offset=0)
        except KeyError as exc:
            raise ValueError(
                f"Invalid alias {alias_name!r} on {cls.__name__}: {path!r}"
            ) from exc

    @classmethod
    def from_array(cls, data: FloatArray) -> Self:
        return cls(data)

    @classmethod
    def zeros(cls) -> Self:
        obj = cls(np.zeros(cls.SIZE, dtype=np.float64))
        obj.apply_defaults()
        return obj

    @classmethod
    def batch_from_array(cls, data: FloatArray) -> BatchArrayView[Self]:
        return BatchArrayView(data, cls)

    @classmethod
    def batch_zeros(cls, count: int) -> BatchArrayView[Self]:
        batch = BatchArrayView(
            np.zeros((count, cls.SIZE), dtype=np.float64),
            cls,
        )

        for row in batch.data:
            cls.from_array(row).apply_defaults()

        return batch

    def copy(self) -> Self:
        return type(self).from_array(self.data.copy())

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.data[key] = value

    def apply_defaults(self) -> None:
        for name, spec in self.FIELD_SPECS.items():
            if not isinstance(spec, int):
                nested = getattr(self, name)
                nested.apply_defaults()

        for path, value in self.__defaults__.items():
            target = self._resolve_path(path)
            target[...] = value

    def _resolve_path(self, path: str) -> Any:
        obj: Any = self
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    @classmethod
    def slice_of(cls, path: str) -> slice:
        path = cls.__aliases__.get(path, path)
        parts = path.split(".")
        return cls._slice_of_parts(parts, base_offset=0)

    @classmethod
    def _slice_of_parts(cls, parts: list[str], base_offset: int) -> slice:
        if not parts:
            return slice(base_offset, base_offset + cls.SIZE)

        part = parts[0]

        if part in cls.__aliases__:
            alias_parts = cls.__aliases__[part].split(".")
            return cls._slice_of_parts(alias_parts + parts[1:], base_offset)

        if part not in cls.SLICES:
            raise KeyError(
                f"{part!r} is not a field of {cls.__name__}. "
                f"Available fields: {list(cls.SLICES)}. "
                f"Available aliases: {list(cls.__aliases__)}."
            )

        local_slice = cls.SLICES[part]
        spec = cls.FIELD_SPECS[part]

        start = base_offset + local_slice.start
        stop = base_offset + local_slice.stop

        if len(parts) == 1:
            return slice(start, stop)

        if isinstance(spec, int):
            raise KeyError(
                f"{part!r} is a raw vector field, so it has no nested field "
                f"{'.'.join(parts[1:])!r}."
            )

        return spec._slice_of_parts(parts[1:], base_offset=start)

    def array_of(self, path: str) -> FloatArray:
        return self.data[self.slice_of(path)]


class BatchArrayView(Generic[TArrayView]):
    __slots__ = ("data", "view_type")

    data: FloatArray
    view_type: type[TArrayView]

    def __init__(self, data: FloatArray, view_type: type[TArrayView]):
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        if data.shape[1] != view_type.SIZE:
            raise ValueError(
                f"Expected second dimension {view_type.SIZE}, "
                f"got {data.shape[1]}"
            )

        self.data = data
        self.view_type = view_type

    def __len__(self) -> int:
        return self.data.shape[0]

    @property
    def count(self) -> int:
        return self.data.shape[0]

    @overload
    def __getitem__(self, idx: int) -> TArrayView:
        ...

    @overload
    def __getitem__(self, idx: slice) -> BatchArrayView[TArrayView]:
        ...

    def __getitem__(self, idx: int | slice) -> TArrayView | BatchArrayView[TArrayView]:
        rows = self.data[idx]

        if rows.ndim == 1:
            return self.view_type.from_array(rows)

        return type(self)(rows, self.view_type)

    def __iter__(self) -> Iterator[TArrayView]:
        for i in range(len(self)):
            yield self.view_type.from_array(self.data[i])

    def copy(self) -> BatchArrayView[TArrayView]:
        return type(self)(self.data.copy(), self.view_type)

    def array_of(self, path: str) -> FloatArray:
        sl = self.view_type.slice_of(path)
        return self.data[:, sl]

    def __getattr__(self, name: str) -> FloatArray:
        if name in self.view_type.SLICES or name in self.view_type.__aliases__:
            return self.array_of(name)

        raise AttributeError(
            f"{type(self).__name__} object has no attribute {name!r}"
        )
