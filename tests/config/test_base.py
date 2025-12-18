from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.config.base import (
    get_all_registered,
    get_field_mappings,
    get_section_name,
    parse_config,
    register_config,
)


def test_register_config_decorator():
    @register_config(section_name="test_section")
    @dataclass
    class NewConfig:
        setting: str = "test"

    assert NewConfig in get_all_registered()
    assert get_section_name(NewConfig) == "test_section"


def test_register_config_with_field_mappings():
    @register_config(section_name="mapped", field_mappings={"internal_name": "external_key"})
    @dataclass
    class MappedConfig:
        internal_name: str = "value"

    mappings = get_field_mappings(MappedConfig)
    assert mappings["internal_name"] == "external_key"


def test_register_config_with_parsers():
    def custom_parser(value: str) -> int:
        return int(value) * 2

    @register_config(section_name="parsed", field_parsers={"doubled": custom_parser})
    @dataclass
    class ParsedConfig:
        doubled: int = 10

    section_data = {"doubled": "5"}
    config = parse_config(ParsedConfig, section_data)
    assert config.doubled == 10


@pytest.mark.parametrize(
    "section_data, expected_name, expected_count, expected_enabled",
    [
        ({"name": "test", "count": 42, "enabled": True}, "test", 42, True),
        ({"name": "prod", "count": 100, "enabled": False}, "prod", 100, False),
        ({"name": "", "count": 0, "enabled": True}, "", 0, True),
    ],
)
def test_parse_config_basic(
    section_data: dict, expected_name: str, expected_count: int, expected_enabled: bool
):
    @register_config(section_name="basic_test")
    @dataclass
    class BasicConfig:
        name: str = "default"
        count: int = 0
        enabled: bool = False

    config = parse_config(BasicConfig, section_data)

    assert config.name == expected_name
    assert config.count == expected_count
    assert config.enabled == expected_enabled


@pytest.mark.parametrize(
    "section_data, expected_required, expected_optional",
    [
        ({"required": "provided"}, "provided", 100),
        ({"required": "test", "optional": 200}, "test", 200),
        ({"required": "value", "optional": 0}, "value", 0),
    ],
)
def test_parse_config_with_defaults(
    section_data: dict, expected_required: str, expected_optional: int
):
    @register_config(section_name="defaults_test")
    @dataclass
    class DefaultsConfig:
        required: str = "default_value"
        optional: int = 100

    config = parse_config(DefaultsConfig, section_data)

    assert config.required == expected_required
    assert config.optional == expected_optional


@pytest.mark.parametrize(
    "section_data, expected_output, expected_level",
    [
        ({"log_output": "file", "log_level": "DEBUG"}, "file", "DEBUG"),
        ({"log_output": "stdout", "log_level": "INFO"}, "stdout", "INFO"),
        ({"log_output": "both", "log_level": "ERROR"}, "both", "ERROR"),
    ],
)
def test_parse_config_with_field_mappings(
    section_data: dict, expected_output: str, expected_level: str
):
    @register_config(
        section_name="mappings_test", field_mappings={"output": "log_output", "level": "log_level"}
    )
    @dataclass
    class MappingsConfig:
        output: str = "stdout"
        level: str = "INFO"

    config = parse_config(MappingsConfig, section_data)

    assert config.output == expected_output
    assert config.level == expected_level


@pytest.mark.parametrize(
    "section_data, expected_name",
    [({"name": "hello"}, "HELLO"), ({"name": "world"}, "WORLD"), ({"name": "TeSt"}, "TEST")],
)
def test_parse_config_with_custom_parser(section_data: dict, expected_name: str):
    def parse_upper(value: str) -> str:
        return value.upper()

    @register_config(section_name="custom_test", field_parsers={"name": parse_upper})
    @dataclass
    class CustomConfig:
        name: str = "default"

    config = parse_config(CustomConfig, section_data)
    assert config.name == expected_name


@pytest.mark.parametrize(
    "section_data, expected_data_dir, expected_output_dir",
    [
        (
            {"data_dir": "custom/data", "output_dir": "custom/output"},
            Path("custom/data"),
            Path("custom/output"),
        ),
        ({"data_dir": "test", "output_dir": "out"}, Path("test"), Path("out")),
        ({"data_dir": "/abs/path", "output_dir": "/abs/out"}, Path("/abs/path"), Path("/abs/out")),
    ],
)
def test_parse_config_path_conversion(
    section_data: dict, expected_data_dir: Path, expected_output_dir: Path
):
    @register_config(section_name="paths_test")
    @dataclass
    class PathConfig:
        data_dir: Path = Path("default/path")
        output_dir: Path = Path("output")

    config = parse_config(PathConfig, section_data)

    assert isinstance(config.data_dir, Path)
    assert isinstance(config.output_dir, Path)
    assert config.data_dir == expected_data_dir
    assert config.output_dir == expected_output_dir


@pytest.mark.parametrize(
    "section_data, expected_items",
    [
        ({"items": ["a", "b", "c"]}, ["a", "b", "c"]),
        ({"items": []}, []),
        ({"items": ["single"]}, ["single"]),
        ({"items": ["1", "2", "3", "4", "5"]}, ["1", "2", "3", "4", "5"]),
    ],
)
def test_parse_config_list_field(section_data: dict, expected_items: list):
    @register_config(section_name="lists_test")
    @dataclass
    class ListConfig:
        items: list[str] = field(default_factory=list)

    config = parse_config(ListConfig, section_data)
    assert config.items == expected_items


def test_parse_config_unregistered_class():
    @dataclass
    class UnregisteredConfig:
        value: str = "test"

    with pytest.raises(ValueError, match="not registered"):
        parse_config(UnregisteredConfig, {"value": "data"})


def test_get_section_name_unregistered():
    @dataclass
    class UnregisteredConfig:
        value: str = "test"

    with pytest.raises(ValueError, match="not registered"):
        get_section_name(UnregisteredConfig)


def test_get_field_mappings_unregistered():
    @dataclass
    class UnregisteredConfig:
        value: str = "test"

    with pytest.raises(ValueError, match="not registered"):
        get_field_mappings(UnregisteredConfig)


def test_get_all_registered_contains_actual_configs():
    from src.config import ImageProcessingConfig, LoggerConfig

    registered = get_all_registered()

    assert LoggerConfig in registered
    assert ImageProcessingConfig in registered


@pytest.mark.parametrize(
    "num_value, text_value, expected_num, expected_text",
    [(5, "hello", 10, "HELLO"), (10, "world", 20, "WORLD"), (0, "test", 0, "TEST")],
)
def test_multiple_field_mappings_and_parsers(
    num_value: int, text_value: str, expected_num: int, expected_text: str
):
    """Test config with multiple field mappings and parsers."""

    def double(x: int) -> int:
        return x * 2

    def uppercase(s: str) -> str:
        return s.upper()

    @register_config(
        section_name="complex_test",
        field_mappings={"internal_num": "num", "internal_text": "text"},
        field_parsers={"internal_num": double, "internal_text": uppercase},
    )
    @dataclass
    class ComplexConfig:
        internal_num: int = 0
        internal_text: str = ""

    section_data = {"num": num_value, "text": text_value}
    config = parse_config(ComplexConfig, section_data)

    assert config.internal_num == expected_num
    assert config.internal_text == expected_text
