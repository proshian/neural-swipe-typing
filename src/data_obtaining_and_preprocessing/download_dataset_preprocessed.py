import gdown

if __name__ == "__main__":
    DATA_PATH = "../data/data_preprocessed"
    
    url = "https://drive.google.com/drive/folders/1V2QxYfxkqHnMM3I-OJjYzlP5AgmMyiAN"
    
    gdown.download_folder(url,
                          output=DATA_PATH,
                          quiet=False,
                          use_cookies=False)
