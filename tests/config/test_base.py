from dataclasses import dataclass

import pytest

from src.config.base import (
    get_all_registered,
    get_field_mappings,
    get_section_name,
    register_config,
)


def test_register_config_decorator():
    @register_config(name="test_section")
    @dataclass
    class NewConfig:
        setting: str = "test"

    assert NewConfig in get_all_registered()
    assert get_section_name(NewConfig) == "test_section"


def test_register_config_with_field_mappings():
    @register_config(name="mapped", field_mappings={"internal_name": "external_key"})
    @dataclass
    class MappedConfig:
        internal_name: str = "value"

    mappings = get_field_mappings(MappedConfig)
    assert mappings["internal_name"] == "external_key"


def test_register_config_with_parsers():
    from src.config.parser import ConfigParser

    def custom_parser(value: str, _parser: ConfigParser) -> int:
        return int(value) * 2

    @register_config(name="parsed", field_parsers={"doubled": custom_parser})
    @dataclass
    class ParsedConfig:
        doubled: int = 10

    parser = ConfigParser()
    section_data = {"doubled": "5"}
    config = parser.parse_config(ParsedConfig, section_data)
    assert config.doubled == 10


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
    from src.image_processing import ImageProcessingConfig
    from src.utils.logger import LoggerConfig

    registered = get_all_registered()

    assert LoggerConfig in registered
    assert ImageProcessingConfig in registered
