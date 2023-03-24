import re
import pandas as pd

pattern = r'([a-zA-Z]+): (.*)'

data = {
    'name': [],
    'line': []
}

with open('data/cbbb.txt') as f:
    for line in f:
        match = re.findall(pattern, line)
        if match:
            name, line = match[0]
            if ":" in line:
                continue
            data['name'].append(name)
            data['line'].append(line)

df = pd.DataFrame(data)
df.to_csv('data/cbbb.csv', index=False)
