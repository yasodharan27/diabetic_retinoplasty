"""
Lightweight tests for config.py's preprocessing configuration surface
(PreprocessingConfig, PREPROCESSING, _env_float/_env_int). Verifies the
environment-variable-override mechanism itself, isolated from
image_preprocessing.py (which is covered separately in
tests/test_image_preprocessing.py).
"""

import importlib
import os
import unittest

import config as config_module

_ENV_KEYS = (
    "DEFAULT_GAMMA",
    "DEFAULT_CLAHE_CLIP_LIMIT",
    "DEFAULT_CLAHE_TILE_GRID_WIDTH",
    "DEFAULT_CLAHE_TILE_GRID_HEIGHT",
)


class EnvParsingHelperTests(unittest.TestCase):
    def test_env_float_uses_default_when_unset(self):
        os.environ.pop("DOES_NOT_EXIST_FLOAT", None)
        self.assertEqual(config_module._env_float("DOES_NOT_EXIST_FLOAT", 1.2), 1.2)

    def test_env_float_parses_set_value(self):
        os.environ["DOES_NOT_EXIST_FLOAT"] = "3.3"
        try:
            self.assertEqual(config_module._env_float("DOES_NOT_EXIST_FLOAT", 1.2), 3.3)
        finally:
            os.environ.pop("DOES_NOT_EXIST_FLOAT", None)

    def test_env_int_uses_default_when_unset(self):
        os.environ.pop("DOES_NOT_EXIST_INT", None)
        self.assertEqual(config_module._env_int("DOES_NOT_EXIST_INT", 8), 8)

    def test_env_int_parses_set_value(self):
        os.environ["DOES_NOT_EXIST_INT"] = "16"
        try:
            self.assertEqual(config_module._env_int("DOES_NOT_EXIST_INT", 8), 16)
        finally:
            os.environ.pop("DOES_NOT_EXIST_INT", None)


class PreprocessingConfigDefaultsTests(unittest.TestCase):
    def test_defaults_match_documented_sensible_values(self):
        # These are the values documented in .env_sample; also pins the
        # numeric defaults so accidental drift breaks this test loudly.
        self.assertEqual(config_module.PREPROCESSING.DEFAULT_GAMMA, 1.2)
        self.assertEqual(config_module.PREPROCESSING.DEFAULT_CLAHE_CLIP_LIMIT, 2.0)
        self.assertEqual(config_module.PREPROCESSING.DEFAULT_CLAHE_TILE_GRID_SIZE, (8, 8))


class PreprocessingConfigEnvironmentOverrideTests(unittest.TestCase):
    """Integration test: proves PREPROCESSING is actually built from the
    environment variables listed in .env_sample, via a real module reload
    (config.py reads os.environ once at import time, same as every other
    setting in that file)."""

    def setUp(self):
        self._env_backup = {key: os.environ.get(key) for key in _ENV_KEYS}

    def tearDown(self):
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        importlib.reload(config_module)

    def test_environment_variables_override_preprocessing_defaults(self):
        os.environ["DEFAULT_GAMMA"] = "3.3"
        os.environ["DEFAULT_CLAHE_CLIP_LIMIT"] = "5.5"
        os.environ["DEFAULT_CLAHE_TILE_GRID_WIDTH"] = "4"
        os.environ["DEFAULT_CLAHE_TILE_GRID_HEIGHT"] = "16"

        importlib.reload(config_module)

        self.assertEqual(config_module.PREPROCESSING.DEFAULT_GAMMA, 3.3)
        self.assertEqual(config_module.PREPROCESSING.DEFAULT_CLAHE_CLIP_LIMIT, 5.5)
        self.assertEqual(config_module.PREPROCESSING.DEFAULT_CLAHE_TILE_GRID_SIZE, (4, 16))


if __name__ == "__main__":
    unittest.main()
