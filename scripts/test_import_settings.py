import sys
sys.path.insert(0, 'c:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/AI4WIND/CENOP')
from cenop.ui.tabs import settings
print('Loaded settings OK')
panel = settings._data_available_panel()
print('Panel type:', type(panel))
print('Panel repr snippet:', repr(panel)[:200])
