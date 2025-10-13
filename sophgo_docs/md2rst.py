import requests
from pathlib import Path

def convert(from_file, dst_file):
    print(f'Convert from {from_file} to {dst_file}...')
    response = requests.post(
        url = 'http://c.docverter.com/convert',
        data = {'to': 'rst', 'from': 'markdown'},
        files = {'input_files[]': open(from_file, 'rb')}
    )

    if response.ok:
        with open(dst_file, 'wb') as f:
            f.write(response.content)
        print(f'Convert Success, dst file in: {dst_file}')
    else:
        print(f'Convert Failed, error code: {response.status_code}')

if __name__ == '__main__':
    CUR_PATH = Path(__file__).resolve().parent
    ROOT_PATH = CUR_PATH.parent
    ROOT_README = Path.joinpath(ROOT_PATH, 'README.md')
    SOPH_DOC = Path.joinpath(CUR_PATH, 'source_zh', 'readme.rst')
    # print(f'{CUR_PATH}, {ROOT_PATH}, {ROOT_README}, {SOPH_DOC}')
    convert(ROOT_README, SOPH_DOC)