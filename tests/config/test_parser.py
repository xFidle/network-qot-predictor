from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from src.config import ConfigParser, ImageProcessingConfig, LoggerConfig
from src.config.base import register_config


def test_init_creates_example_config_if_missing(tmp_path: Path):
    config_file = tmp_path / "test_config.toml"
    assert not config_file.exists()

    parser = ConfigParser(config_file)

    assert config_file.exists()
    assert parser._config is not None


def test_init_loads_existing_config(tmp_path: Path):
    config_file = tmp_path / "existing.toml"
    config_file.write_text(
        """
[logging]
log_level = "DEBUG"
log_output = ["stdout"]
log_format = "%(message)s"

[image_processing]
model = "vgg16"
data_dir = "test/data"
force_download = true
"""
    )

    parser = ConfigParser(config_file)

    assert "logging" in parser._config
    assert "image_processing" in parser._config
    assert parser._config["logging"]["log_level"] == "DEBUG"
    assert parser._config["image_processing"]["model"] == "vgg16"


def test_get_generic_method(tmp_path: Path):
    config_file = tmp_path / "test.toml"
    config_file.write_text(
        """
[logging]
level = "ERROR"
output = ["stdout"]
format = "%(message)s"

[image_processing]
model = "resnet50"
data_dir = "data"
"""
    )

    parser = ConfigParser(config_file)

    logger_config = parser.get(LoggerConfig)
    assert isinstance(logger_config, LoggerConfig)
    assert logger_config.level == "ERROR"

    img_config = parser.get(ImageProcessingConfig)
    assert isinstance(img_config, ImageProcessingConfig)
    assert img_config.model == "resnet50"


def test_missing_section_raises_error(tmp_path: Path):
    config_file = tmp_path / "test.toml"
    config_file.write_text(
        """
[image_processing]
model = "resnet50"
data_dir = "data"
"""
    )

    parser = ConfigParser(config_file)

    with pytest.raises(ValueError, match="Missing \\[logging\\] section"):
        parser.get(LoggerConfig)


def test_example_config_has_correct_defaults(tmp_path: Path):
    config_file = tmp_path / "example.toml"
    parser = ConfigParser(config_file)

    default_logger_config = LoggerConfig()
    default_img_config = ImageProcessingConfig()

    logger_config = parser.get(LoggerConfig)
    img_config = parser.get(ImageProcessingConfig)

    assert logger_config.level == default_logger_config.level
    assert logger_config.output == default_logger_config.output
    assert logger_config.format_string == default_logger_config.format_string

    assert img_config.model == default_img_config.model
    assert img_config.data_dir == default_img_config.data_dir
    assert img_config.force_download is default_img_config.force_download


def test_field_mappings_work_in_parser(tmp_path: Path):
    config_file = tmp_path / "test.toml"
    config_file.write_text(
        """
[logging]
output = ["file"]
level = "DEBUG"
format_string = "custom format"
file = "custom.log"

[image_processing]
model = "resnet50"
data_dir = "data"
"""
    )

    parser = ConfigParser(config_file)
    config = parser.get(LoggerConfig)

    assert config.output == ["file"]
    assert config.level == "DEBUG"
    assert config.format_string == "custom format"


@pytest.mark.parametrize("data_dir", ["some/string/path", "relative/path", "/absolute/path"])
def test_path_string_conversion(tmp_path, data_dir: str):
    config_file = tmp_path / "test.toml"
    config_file.write_text(
        f"""
[logging]
log_level = "INFO"
log_output = ["stdout"]
log_format = "%(message)s"

[image_processing]
model = "resnet50"
data_dir = "{data_dir}"
"""
    )

    parser = ConfigParser(config_file)
    config = parser.get(ImageProcessingConfig)

    assert isinstance(config.data_dir, Path)
    assert str(config.data_dir) == data_dir


def test_config_path_accepts_string(tmp_path: Path):
    config_file = tmp_path / "test.toml"
    parser = ConfigParser(str(config_file))

    assert parser._config_path == Path(config_file)
    assert config_file.exists()


def test_config_path_accepts_path_object(tmp_path: Path):
    config_file = tmp_path / "test.toml"
    parser = ConfigParser(config_file)

    assert parser._config_path == config_file
    assert config_file.exists()


@pytest.mark.parametrize("force_download, expected", [(True, True), (False, False)])
def test_boolean_values_parsed_correctly(tmp_path: Path, force_download: bool, expected: bool):
    config_file = tmp_path / "test.toml"
    config_file.write_text(
        f"""
[logging]
log_level = "INFO"
log_output = ["stdout"]
log_format = "%(message)s"

[image_processing]
model = "resnet50"
data_dir = "data"
force_download = {str(force_download).lower()}
"""
    )

    parser = ConfigParser(config_file)
    config = parser.get(ImageProcessingConfig)

    assert config.force_download is expected
    assert isinstance(config.force_download, bool)


@pytest.mark.parametrize(
    "log_output, expected",
    [(["stdout", "file"], ["stdout", "file"]), (["stdout"], ["stdout"]), (["file"], ["file"])],
)
def test_list_values_parsed_correctly(tmp_path: Path, log_output: list[str], expected: list[str]):
    config_file = tmp_path / "test.toml"
    config_file.write_text(
        f"""
[logging]
level = "INFO"
output = {log_output}
format = "%(message)s"

[image_processing]
model = "resnet50"
data_dir = "data"
"""
    )

    parser = ConfigParser(config_file)
    config = parser.get(LoggerConfig)

    assert config.output == expected
    assert isinstance(config.output, list)


@pytest.mark.parametrize(
    "section_data, expected_name, expected_count, expected_enabled",
    [
        ({"name": "test", "count": 42, "enabled": True}, "test", 42, True),
        ({"name": "prod", "count": 100, "enabled": False}, "prod", 100, False),
        ({"name": "", "count": 0, "enabled": True}, "", 0, True),
    ],
)
def test_parse_config_basic(
    section_data: dict[str, Any], expected_name: str, expected_count: int, expected_enabled: bool
):
    @register_config(name="basic_test")
    @dataclass
    class BasicConfig:
        name: str = "default"
        count: int = 0
        enabled: bool = False

    parser = ConfigParser()
    config = parser.parse_config(BasicConfig, section_data)

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
    section_data: dict[str, Any], expected_required: str, expected_optional: int
):
    @register_config(name="defaults_test")
    @dataclass
    class DefaultsConfig:
        required: str = "default_value"
        optional: int = 100

    parser = ConfigParser()
    config = parser.parse_config(DefaultsConfig, section_data)

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
    section_data: dict[str, Any], expected_output: str, expected_level: str
):
    @register_config(
        name="mappings_test", field_mappings={"output": "log_output", "level": "log_level"}
    )
    @dataclass
    class MappingsConfig:
        output: str = "stdout"
        level: str = "INFO"

    parser = ConfigParser()
    config = parser.parse_config(MappingsConfig, section_data)

    assert config.output == expected_output
    assert config.level == expected_level


@pytest.mark.parametrize(
    "section_data, expected_name",
    [({"name": "hello"}, "HELLO"), ({"name": "world"}, "WORLD"), ({"name": "TeSt"}, "TEST")],
)
def test_parse_config_with_custom_parser(section_data: dict[str, Any], expected_name: str):
    def parse_upper(value: str, _: ConfigParser) -> str:
        return value.upper()

    @register_config(name="custom_test", field_parsers={"name": parse_upper})
    @dataclass
    class CustomConfig:
        name: str = "default"

    parser = ConfigParser()
    config = parser.parse_config(CustomConfig, section_data)
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
    section_data: dict[str, Any], expected_data_dir: Path, expected_output_dir: Path
):
    @register_config(name="paths_test")
    @dataclass
    class PathConfig:
        data_dir: Path = Path("default/path")
        output_dir: Path = Path("output")

    parser = ConfigParser()
    config = parser.parse_config(PathConfig, section_data)

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
def test_parse_config_list_field(section_data: dict[str, Any], expected_items: list[str]):
    @register_config(name="lists_test")
    @dataclass
    class ListConfig:
        items: list[str] = field(default_factory=list)

    parser = ConfigParser()
    config = parser.parse_config(ListConfig, section_data)
    assert config.items == expected_items


def test_parse_config_unregistered_class():
    @dataclass
    class UnregisteredConfig:
        value: str = "test"

    parser = ConfigParser()
    with pytest.raises(ValueError, match="not registered"):
        parser.parse_config(UnregisteredConfig, {"value": "data"})


@pytest.mark.parametrize(
    "num_value, text_value, expected_num, expected_text",
    [(5, "hello", 10, "HELLO"), (10, "world", 20, "WORLD"), (0, "test", 0, "TEST")],
)
def test_multiple_field_mappings_and_parsers(
    num_value: int, text_value: str, expected_num: int, expected_text: str
):
    def double(x: int, _: ConfigParser) -> int:
        return x * 2

    def uppercase(s: str, _: ConfigParser) -> str:
        return s.upper()

    @register_config(
        name="complex_test",
        field_mappings={"internal_num": "num", "internal_text": "text"},
        field_parsers={"internal_num": double, "internal_text": uppercase},
    )
    @dataclass
    class ComplexConfig:
        internal_num: int = 0
        internal_text: str = ""

    section_data = {"num": num_value, "text": text_value}
    parser = ConfigParser()
    config = parser.parse_config(ComplexConfig, section_data)

    assert config.internal_num == expected_num
    assert config.internal_text == expected_text


@pytest.mark.parametrize(
    "section_data, expected_path, expected_name",
    [
        ({"data_path": "some/path", "name": "test"}, Path("some/path"), "test"),
        ({"name": "only_name"}, None, "only_name"),
        ({"data_path": "/abs/path"}, Path("/abs/path"), "default"),
    ],
)
def test_parse_config_optional_path(
    section_data: dict, expected_path: Path | None, expected_name: str
):
    @register_config(name="optional_test")
    @dataclass
    class OptionalConfig:
        data_path: Path | None = None
        name: str = "default"

    parser = ConfigParser()
    config = parser.parse_config(OptionalConfig, section_data)

    assert config.data_path == expected_path
    if expected_path is not None:
        assert isinstance(config.data_path, Path)
    assert config.name == expected_name


def test_parse_config_nested_dataclass():
    @register_config(name="nested")
    @dataclass
    class NestedConfig:
        host: str = "localhost"
        port: int = 8080

    @register_config(name="parent")
    @dataclass
    class ParentConfig:
        name: str = "app"
        server: NestedConfig = field(default_factory=lambda: NestedConfig())

    section_data = {"name": "myapp", "server": {"host": "example.com", "port": 9000}}

    parser = ConfigParser()
    config = parser.parse_config(ParentConfig, section_data)

    assert config.name == "myapp"
    assert isinstance(config.server, NestedConfig)
    assert config.server.host == "example.com"
    assert config.server.port == 9000


def test_parse_config_optional_nested_dataclass():
    @register_config(name="db_config")
    @dataclass
    class DatabaseConfig:
        host: str = "localhost"
        port: int = 5432

    @register_config(name="app_config")
    @dataclass
    class AppConfig:
        name: str = "app"
        database: DatabaseConfig | None = None

    section_data_with_db = {"name": "myapp", "database": {"host": "db.example.com", "port": 3306}}
    parser = ConfigParser()
    config_with_db = parser.parse_config(AppConfig, section_data_with_db)

    assert config_with_db.name == "myapp"
    assert isinstance(config_with_db.database, DatabaseConfig)
    assert config_with_db.database.host == "db.example.com"
    assert config_with_db.database.port == 3306

    section_data_no_db = {"name": "simpleapp"}
    config_no_db = parser.parse_config(AppConfig, section_data_no_db)

    assert config_no_db.name == "simpleapp"
    assert config_no_db.database is None


def test_field_serializers_basic(tmp_path: Path):
    @register_config(name="serializer_test", field_serializers={"upper_name": lambda x: x.lower()})
    @dataclass
    class _SerializerConfig:
        upper_name: str = "DEFAULT"
        count: int = 42

    config_file = tmp_path / "test.toml"
    _parser = ConfigParser(config_file)

    import tomllib

    with open(config_file, "rb") as f:
        data = tomllib.load(f)

    assert "serializer_test" in data
    assert data["serializer_test"]["upper_name"] == "default"
    assert data["serializer_test"]["count"] == 42


def test_field_serializers_with_path(tmp_path: Path):
    class CustomPath:
        def __init__(self, path: Path):
            self.path = path

        def __str__(self):
            return f"custom://{self.path}"

    @register_config(name="custom_path_test", field_serializers={"custom_path": lambda x: str(x)})
    @dataclass
    class _CustomPathConfig:
        custom_path: CustomPath = field(default_factory=lambda: CustomPath(Path("default")))
        normal_path: Path = Path("data")

    config_file = tmp_path / "test.toml"
    _parser = ConfigParser(config_file)

    import tomllib

    with open(config_file, "rb") as f:
        data = tomllib.load(f)

    assert data["custom_path_test"]["custom_path"] == "custom://default"
    assert data["custom_path_test"]["normal_path"] == "data"


def test_field_serializers_preserves_none(tmp_path: Path):
    """Test that None values are still skipped even with serializers."""

    @register_config(
        name="none_test", field_serializers={"optional_value": lambda x: x.upper() if x else None}
    )
    @dataclass
    class _NoneConfig:
        required: str = "test"
        optional_value: str | None = None

    config_file = tmp_path / "test.toml"
    _parser = ConfigParser(config_file)

    import tomllib

    with open(config_file, "rb") as f:
        data = tomllib.load(f)

    assert "optional_value" not in data["none_test"]
    assert data["none_test"]["required"] == "test"


def test_parser_can_access_other_configs(tmp_path: Path):
    @register_config(name="base_config")
    @dataclass
    class BaseConfig:
        multiplier: int = 2

    @register_config(
        name="dependent_config",
        field_parsers={"value": lambda x, parser: x * parser.get(BaseConfig).multiplier},
    )
    @dataclass
    class DependentConfig:
        value: int = 10

    config_file = tmp_path / "test.toml"
    config_file.write_text(
        """
[base_config]
multiplier = 3

[dependent_config]
value = 5
"""
    )

    parser = ConfigParser(config_file)
    config = parser.get(DependentConfig)

    assert config.value == 15
