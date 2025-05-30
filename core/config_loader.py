# offline_chat_bot/core/config_loader.py

import yaml
import os

_config = None
_project_root = None

def get_project_root() -> str:
    """Determines and returns the project root directory."""
    global _project_root
    if _project_root is None:
        # Assuming this file (config_loader.py) is in offline_chat_bot/core/
        # So, ../../ goes up to offline_chat_bot
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return _project_root

def load_config(config_path: str = None) -> dict:
    """
    Loads the YAML configuration file.
    Caches the loaded configuration.

    Args:
        config_path (str, optional): Absolute path to the config.yaml file.
                                     If None, defaults to 'config.yaml' in project root.

    Returns:
        dict: The loaded configuration.
    """
    global _config
    if _config is not None:
        return _config

    if config_path is None:
        root_dir = get_project_root()
        config_path = os.path.join(root_dir, 'config.yaml')

    try:
        with open(config_path, 'r') as f:
            _config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from: {config_path}")
        
        # Ensure data directory exists based on config
        data_dir_name = _config.get('data_dir', 'data') # Default to 'data' if not in config
        abs_data_dir = os.path.join(get_project_root(), data_dir_name)
        if not os.path.exists(abs_data_dir):
            os.makedirs(abs_data_dir)
            print(f"Created data directory: {abs_data_dir}")
        
        return _config
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"ERROR: Error parsing YAML configuration file {config_path}: {e}")
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error loading configuration: {e}")
        raise

def get_config() -> dict:
    """
    Returns the loaded configuration. Loads it if not already loaded.
    This is the primary way other modules should get the config.
    """
    if _config is None:
        return load_config() # Load with default path
    return _config

# --- Test Block for config_loader ---
if __name__ == "__main__":
    print("--- Testing Configuration Loader ---")
    
    # Test loading with default path
    try:
        cfg = get_config()
        if cfg:
            print("\nSuccessfully loaded configuration (first call):")
            # Print some sample values
            print(f"  LSW Host: {cfg.get('lsw', {}).get('ollama_host')}")
            print(f"  MMU STM Max Turns: {cfg.get('mmu', {}).get('stm_max_turns')}")
            print(f"  Orchestrator Target Tokens: {cfg.get('orchestrator', {}).get('target_max_prompt_tokens')}")
            print(f"  Data directory from config: {cfg.get('data_dir', 'data')}")
            project_r = get_project_root()
            print(f"  Project root: {project_r}")
            print(f"  Absolute MTM DB Path: {os.path.join(project_r, cfg.get('mmu',{}).get('mtm_db_path',''))}")
            
            # Test caching (second call should not print "Configuration loaded successfully...")
            print("\nAttempting to get config again (should use cache):")
            cfg_cached = get_config()
            if cfg_cached is cfg: # Check if it's the same object
                print("  Successfully retrieved cached configuration.")
            else:
                print("  ERROR: Configuration was reloaded instead of using cache.")
        else:
            print("Configuration loading returned None/Empty.")
            
    except Exception as e:
        print(f"Test failed: {e}")