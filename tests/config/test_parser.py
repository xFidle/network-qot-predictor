from pathlib import Path

import pytest

from src.config import ConfigParser, ImageProcessingConfig, LoggerConfig


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
