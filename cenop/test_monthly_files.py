import sys
sys.path.insert(0, 'src')
from cenop.landscape.loader import LandscapeLoader

landscapes = ['CentralBaltic', 'DanTysk', 'Gemini', 'Kattegat', 'NorthSea', 'UserDefined']

for name in landscapes:
    try:
        loader = LandscapeLoader(name)
        info = loader.list_files()
        prey_count = len(info.get('prey_months', []))
        sal_count = len(info.get('salinity_months', []))
        print(f'{name:15s}: prey={prey_count:2d}/12  sal={sal_count:2d}/12')
    except Exception as e:
        print(f'{name:15s}: ERROR - {e}')
