
import sys
import time
from unittest.mock import MagicMock
from contextlib import contextmanager

# Define StateDict at module level
class StateDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        
        # If the key is 'selected_dashboard', return a string
        if key == "selected_dashboard":
            return "Voronoi"

        # Return a string-like mock for unknown keys to satisfy re.sub
        # We need an object that behaves like a string
        class StringMock(str):
            def __new__(cls, *args, **kwargs):
                return str.__new__(cls, "mock")
            def __getattr__(self, name):
                return MagicMock()
            def lower(self): return StringMock()
            def strip(self, *args): return StringMock()
        
        return StringMock()
    def __setattr__(self, key, value):
        self[key] = value

# Mock streamlit before importing the dashboard
class MockStreamlit:
    def __init__(self):
        self._session_state = None
        self._query_params = {}

    def __getattr__(self, name):
        if name == "session_state":
            return self.session_state
        if name == "query_params":
            return self.query_params
        return MagicMock()

    @contextmanager
    def container(self):
        yield
    
    @property
    def sidebar(self):
        return self.container()
    
    @property
    def expander(self, *args, **kwargs):
        return self.container()

    @property
    def spinner(self, *args, **kwargs):
        return self.container()

    def set_page_config(self, *args, **kwargs):
        pass

    def radio(self, label, options, index=0, key=None, on_change=None, **kwargs):
        if key and key in self.session_state:
            return self.session_state[key]
        return options[index]

    @property
    def session_state(self):
        if self._session_state is None:
            # Use a real dict for session state to avoid MagicMock issues with strings
            self._session_state = {"selected_dashboard": "Voronoi"}
            self._session_state = StateDict(self._session_state)
            
        return self._session_state

    @property
    def query_params(self):
        return self._query_params
    
    @query_params.setter
    def query_params(self, value):
        self._query_params = value

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible"):
        return value if value is not None else min_value

    def selectbox(self, label, options, index=0, format_func=None, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible"):
        return options[index]

    def checkbox(self, label, value=False, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible"):
        return value

    def multiselect(self, label, options, default=None, format_func=None, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible", max_selections=None, placeholder="Choose an option"):
        return default if default is not None else []
    
    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible"):
        return value if value is not None else min_value

# Replace the real streamlit with our mock
sys.modules["streamlit"] = MockStreamlit()

# Also need to mock plotly.graph_objects if we want to avoid its import cost? 
# No, the user WANTS to profile import costs. So we keep real imports for everything else.

def profile_load():
    print("Starting profiling of src/integrated_dashboard.py...")
    start_time = time.perf_counter()
    
    # Add src to path
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
    
    # Import the dashboard script
    # This will execute the top-level code (imports, data loading, etc.)
    import integrated_dashboard
    
    end_time = time.perf_counter()
    print(f"Total load time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    profile_load()
