import logging
from pathlib import Path

import pytest

from src.config import ConfigParser, ImageProcessingConfig, LoggerConfig


def test_init_creates_example_config_if_missing(tmp_path):
    config_file = tmp_path / "test_config.toml"
    assert not config_file.exists()

    parser = ConfigParser(config_file)

    assert config_file.exists()
    assert parser._raw_config is not None


def test_init_loads_existing_config(tmp_path):
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
output_dir = "test/output"
force_download = true
"""
    )

    parser = ConfigParser(config_file)

    assert "logging" in parser._raw_config
    assert "image_processing" in parser._raw_config
    assert parser._raw_config["logging"]["log_level"] == "DEBUG"
    assert parser._raw_config["image_processing"]["model"] == "vgg16"


def test_get_generic_method(tmp_path):
    config_file = tmp_path / "test.toml"
    config_file.write_text(
        """
[logging]
log_level = "ERROR"
log_output = ["stdout"]
log_format = "%(message)s"

[image_processing]
model = "resnet50"
data_dir = "data"
output_dir = "output"
"""
    )

    parser = ConfigParser(config_file)

    logger_config = parser.get(LoggerConfig)
    assert isinstance(logger_config, LoggerConfig)
    assert logger_config.level == logging.ERROR

    img_config = parser.get(ImageProcessingConfig)
    assert isinstance(img_config, ImageProcessingConfig)
    assert img_config.model == "resnet50"


def test_get_all(tmp_path):
    config_file = tmp_path / "test.toml"
    config_file.write_text(
        """
[logging]
log_level = "INFO"
log_output = ["stdout"]
log_format = "%(message)s"

[image_processing]
model = "resnet50"
data_dir = "data"
output_dir = "output"
"""
    )

    parser = ConfigParser(config_file)
    logger_config, img_config = parser.get_all()

    assert isinstance(logger_config, LoggerConfig)
    assert isinstance(img_config, ImageProcessingConfig)


def test_missing_section_raises_error(tmp_path):
    config_file = tmp_path / "test.toml"
    config_file.write_text(
        """
[image_processing]
model = "resnet50"
data_dir = "data"
output_dir = "output"
"""
    )

    parser = ConfigParser(config_file)

    with pytest.raises(ValueError, match="Missing \\[logging\\] section"):
        parser.get(LoggerConfig)


@pytest.mark.parametrize("invalid_level", ["INVALID_LEVEL", "invalid", "trace", "verbose", ""])
def test_invalid_log_level_raises_error(tmp_path, invalid_level: str):
    config_file = tmp_path / "test.toml"
    config_file.write_text(
        f"""
[logging]
log_level = "{invalid_level}"
log_output = ["stdout"]
log_format = "%(message)s"

[image_processing]
model = "resnet50"
data_dir = "data"
output_dir = "output"
"""
    )

    parser = ConfigParser(config_file)

    with pytest.raises(ValueError, match="Invalid log level"):
        parser.get(LoggerConfig)


def test_example_config_has_all_sections(tmp_path):
    config_file = tmp_path / "example.toml"
    _ = ConfigParser(config_file)

    content = config_file.read_text()

    assert "[logging]" in content
    assert "[image_processing]" in content


def test_example_config_has_correct_defaults(tmp_path):
    config_file = tmp_path / "example.toml"
    parser = ConfigParser(config_file)

    logger_config = parser.get(LoggerConfig)
    img_config = parser.get(ImageProcessingConfig)

    assert logger_config.level == logging.INFO
    assert logger_config.output == ["stdout"]
    assert logger_config.format_string == "%(levelname)s - %(message)s"

    assert img_config.model == "resnet50"
    assert img_config.data_dir == Path("data/flowers/images")
    assert img_config.output_dir == Path("data/flowers/processed")
    assert img_config.force_download is False


def test_field_mappings_work_in_parser(tmp_path):
    config_file = tmp_path / "test.toml"
    config_file.write_text(
        """
[logging]
log_output = ["file"]
log_level = "DEBUG"
log_format = "custom format"
log_file = "custom.log"

[image_processing]
model = "resnet50"
data_dir = "data"
output_dir = "output"
"""
    )

    parser = ConfigParser(config_file)
    config = parser.get(LoggerConfig)

    assert config.output == ["file"]
    assert config.level == logging.DEBUG
    assert config.format_string == "custom format"


@pytest.mark.parametrize(
    "data_dir, output_dir",
    [
        ("some/string/path", "another/string/path"),
        ("relative/path", "relative/output"),
        ("/absolute/path", "/absolute/output"),
    ],
)
def test_path_string_conversion(tmp_path, data_dir: str, output_dir: str):
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
output_dir = "{output_dir}"
"""
    )

    parser = ConfigParser(config_file)
    config = parser.get(ImageProcessingConfig)

    assert isinstance(config.data_dir, Path)
    assert isinstance(config.output_dir, Path)
    assert str(config.data_dir) == data_dir
    assert str(config.output_dir) == output_dir


def test_config_path_accepts_string(tmp_path):
    config_file = tmp_path / "test.toml"
    parser = ConfigParser(str(config_file))

    assert parser.config_path == Path(config_file)
    assert config_file.exists()


def test_config_path_accepts_path_object(tmp_path):
    config_file = tmp_path / "test.toml"
    parser = ConfigParser(config_file)

    assert parser.config_path == config_file
    assert config_file.exists()


@pytest.mark.parametrize("force_download, expected", [(True, True), (False, False)])
def test_boolean_values_parsed_correctly(tmp_path, force_download: bool, expected: bool):
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
output_dir = "output"
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
def test_list_values_parsed_correctly(tmp_path, log_output: list, expected: list):
    config_file = tmp_path / "test.toml"
    config_file.write_text(
        f"""
[logging]
log_level = "INFO"
log_output = {log_output}
log_format = "%(message)s"

[image_processing]
model = "resnet50"
data_dir = "data"
output_dir = "output"
"""
    )

    parser = ConfigParser(config_file)
    config = parser.get(LoggerConfig)

    assert config.output == expected
    assert isinstance(config.output, list)
