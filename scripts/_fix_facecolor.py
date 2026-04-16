import os
def fix_dir(d):
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith('.ipynb') or file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if "'figure.facecolor': '#0d1117'" in content or '"figure.facecolor": "#0d1117"' in content:
                    content = content.replace("'figure.facecolor': '#0d1117'", "'figure.facecolor': '#0d1117'")
                    content = content.replace('"figure.facecolor": "#0d1117"', '"figure.facecolor": "#0d1117"')
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print('Fixed', path)

fix_dir('notebooks')
fix_dir('scripts')
print('Done scanning')
