import pathlib
p = pathlib.Path('cenop/src/cenop/agents/population.py')
s = p.read_text(encoding='utf-8')
print('lines:', len(s.splitlines()))
for i, line in enumerate(s.splitlines(), start=1):
    if '"""' in line:
        print(i, line)
print('total triple quotes:', s.count('"""'))
# Show the last 80 chars of the file
print('\n--- tail ---')
print(s[-400:])
