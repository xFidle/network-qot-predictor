import types
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin

from src.config.base import (
    DataclassInstance,
    get_all_registered,
    get_field_mappings,
    get_section_name,
    parse_config,
)
from src.config.config_format import ConfigFormat
from src.config.toml_format import TOMLFormat

T = TypeVar("T", bound=DataclassInstance)


class ConfigParser:
    def __init__(self, config_path: Path | str = "config", format: ConfigFormat = TOMLFormat()):
        base_path = Path(config_path)
        self._config_path = base_path.with_suffix(format.extension)
        self._format = format
        self._config = self._read_config()

    def _read_config(self) -> dict[str, dict[str, Any]]:
        config_data = self._format.read(self._config_path)
        if config_data is None:
            return self._create_example_config()

        return config_data

    def _create_example_config(self) -> dict[str, dict[str, Any]]:
        config_data: dict[str, dict[str, Any]] = {}

        for config_class in self._get_root_configs():
            section_name = get_section_name(config_class)
            default_class = config_class()
            config_class_data = ConfigParser._get_class_fields(default_class)
            config_data[section_name] = config_class_data

        self._format.write(self._config_path, config_data)

        return config_data

    @staticmethod
    def _get_class_fields(config_class: DataclassInstance) -> dict[str, Any]:
        class_data: dict[str, Any] = {}

        field_mappings = {}
        try:
            field_mappings = get_field_mappings(config_class.__class__)
        except ValueError:
            pass

        for field in fields(config_class):
            key = field_mappings.get(field.name, field.name)
            value = getattr(config_class, field.name)

            if isinstance(value, bool):
                value = str(value).lower()
            elif isinstance(value, Path):
                value = str(value)
            elif isinstance(value, DataclassInstance):
                value = ConfigParser._get_class_fields(value)
            elif value is None:
                continue

            class_data[key] = value

        return class_data

    def _get_root_configs(self) -> list[type[DataclassInstance]]:
        all_configs = get_all_registered()
        nested_configs = set[DataclassInstance | type[DataclassInstance]]()
        for config_class in all_configs:
            for field in fields(config_class):
                field_type = field.type
                origin = get_origin(field_type)

                if origin is Union or origin is types.UnionType:
                    type_args = [t for t in get_args(field_type) if t is not type(None)]
                    if type_args:
                        field_type = type_args[0]

                if is_dataclass(field_type):
                    nested_configs.add(field_type)

        return [c for c in all_configs if c not in nested_configs]

    def get(self, config_class: type[T]) -> T:
        """
        Generic method to get any registered config.

        Example:
            parser = ConfigParser()
            logger_config = parser.get(LoggerConfig)
            img_config = parser.get(ImageProcessingConfig)
        """
        section_name = get_section_name(config_class)

        if section_name not in self._config:
            raise ValueError(f"Missing [{section_name}] section in config file")

        section_data = self._config[section_name]
        return parse_config(config_class, section_data)
