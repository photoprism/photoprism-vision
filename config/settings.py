import os
import json
from pathlib import Path

class Settings:
    def __init__(self):
        self.config_path = os.getenv('CONFIG_PATH', 'config')
        self.settings_file = Path(self.config_path) / 'settings.json'
        self.settings = self._load_settings()

    def _load_settings(self):
        """Load settings from JSON file"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
            raise  # Raise the exception to notify the caller

        return {}

    def save(self):
        """Save settings to JSON file"""
        if os.getenv('SAVE_SETTINGS', 'false').lower() == 'true':
            try:
                os.makedirs(self.config_path, exist_ok=True)
                with open(self.settings_file, 'w') as f:
                    json.dump(self.settings, f, indent=2)
                return True
            except Exception as e:
                print(f"Error saving settings: {e}")
                raise  # Raise the exception to notify the caller

        return False

    def get(self, key, default=None):
        """Get setting value"""
        return self.settings.get(key, default)

    def set(self, key, value):
        """Set setting value"""
        self.settings[key] = value
        return self.save()
