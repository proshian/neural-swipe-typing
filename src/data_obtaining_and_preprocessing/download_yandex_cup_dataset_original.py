from zipfile import ZipFile
from urllib.parse import urlencode
from io import BytesIO

import requests
from tqdm import tqdm


UNZIP_PATH = '../data/data_original'
CHUNK_SIZE = 4096


def download(url: str, buffer: BytesIO, block_size: int = 1024) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an HTTPError for bad responses
    total_size = int(response.headers.get('content-length', 0))
    
    with tqdm(total=total_size, unit='iB', unit_scale=True, desc='Downloading') as progress_bar:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            buffer.write(data)

    buffer.seek(0)


def extract(buffer: BytesIO, unzip_path: str):
    with ZipFile(buffer) as zf:
        for member in tqdm(zf.infolist(), desc='Extracting'):
            zf.extract(member, unzip_path)
            

def main():
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://disk.yandex.ru/d/IYiSpLob-zAxqg'  # link to file in Yandex.Disk
    final_url = base_url + urlencode(dict(public_key=public_key))

    try:
        response = requests.get(final_url)
        response.raise_for_status()
        download_url = response.json()['href']

        with BytesIO() as buffer:
            download(download_url, buffer, CHUNK_SIZE)
            extract(buffer, UNZIP_PATH)
        
    except requests.RequestException as e:
        print(f"Error downloading or extracting the file: {e}")


if __name__ == '__main__':
    main()