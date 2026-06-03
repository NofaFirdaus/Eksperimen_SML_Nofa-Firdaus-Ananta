import os
import re
import string
import pandas as pd
 
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_PATH    = os.path.join(BASE_DIR, '..', 'hok_reviews.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'hok_preprocessing.csv')
 

 
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)   # hapus mention
    text = re.sub(r'#[A-Za-z0-9]+', '', text)   # hapus hashtag
    text = re.sub(r'RT[\s]', '', text)           # hapus RT
    text = re.sub(r"http\S+", '', text)          # hapus link
    text = re.sub(r'[0-9]+', '', text)           # hapus angka
    text = text.replace('\n', ' ')               # newline → spasi
    text = re.sub(r'[^\w\s]', ' ', text)         # tanda baca → spasi
    text = re.sub(r'\s+', ' ', text).strip()     # hapus spasi berlebih
    return text
 
def casefoldingText(text):
    return text.lower()
 
SLANGWORDS = {
    "@": "di", "gk": "tidak", "ga": "tidak", "tp": "tapi", "jlk": "jelek",
    "gak": "tidak", "y": "iya", "woyyyyy": "halo", "udh": "sudah",
    "udah": "sudah", "dah": "sudah", "gw": "aku", "gua": "aku",
    "yg": "yang", "abis": "habis", "wtb": "beli", "masi": "masih",
    "wts": "jual", "wtt": "tukar", "bgt": "banget", "maks": "maksimal",
    "kalo": "kalau", "kayak": "seperti", "gitu": "begitu", "pake": "pakai",
    "cuman": "cuma", "karna": "karena", "kaya": "seperti", "mulu": "terus",
    "tau": "tahu", "kek": "seperti", "ngelag": "lag", "dapet": "dapat",
    "bener": "benar", "nih": "ini", "ni": "ini", "klo": "kalau",
    "sampe": "sampai", "gimana": "bagaimana", "heronya": "hero",
    "permainan": "main", "bermain": "main", "dikasih": "kasih",
    "nggak": "tidak", "doang": "saja"
}
 
def fix_slangwords(text):
    words = text.split()
    fixed_words = [
        SLANGWORDS[word.lower()] if word.lower() in SLANGWORDS else word
        for word in words
    ]
    return ' '.join(fixed_words)
 
def sentiment_label(score):
    if score <= 2:
        return 'Negatif'
    elif score == 3:
        return 'Netral'
    else:
        return 'Positif'


 
def run_preprocessing(raw_path: str = RAW_PATH,
                      output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    print("=" * 50)
    print("AUTOMATE PREPROCESSING - HOK REVIEWS")
    print("=" * 50)
    # Load
    df = pd.read_csv(raw_path)
    print(f"Dataset dimuat   : {df.shape[0]} baris, {df.shape[1]} kolom")
 
    #sentiment
    df['sentiment'] = df['score'].apply(sentiment_label)

    # Cleaning
    df['text_clean'] = df['content'].apply(cleaningText)
    print(f"cleaningText     : selesai")
 
    # Casefolding
    df['text_casefoldingText'] = df['text_clean'].apply(casefoldingText)
    print(f"casefoldingText  : selesai")
 
    # Slangwords
    df['text_slangwords'] = df['text_casefoldingText'].apply(fix_slangwords)
    print(f"fix_slangwords   : selesai")
 
    # Pilih kolom output untuk training
    df_out = df[['text_slangwords', 'sentiment']].copy()
 
    # Simpan
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Disimpan ke      : {output_path}")
    print(f"[SELESAI] Shape output  : {df_out.shape}")
    print("=" * 50)
 
    return df_out
 
if __name__ == "__main__":
    run_preprocessing()
 