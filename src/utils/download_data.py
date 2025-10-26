import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_and_process(
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META",
           "JPM", "GS", "XOM", "CVX", "JNJ", "UNH",
           "PG", "KO", "SPY"],
    start="2018-01-01",
    end="2023-01-01",
    out_path="data/processed/returns.csv"
):
    """
    Télécharge les prix quotidiens via Yahoo Finance,
    calcule les rendements log et sauvegarde en CSV.
    """
    print(f"⬇️ Téléchargement des données pour {tickers} ...")
    data = yf.download(tickers, start=start, end=end)["Close"]

    print("✅ Données téléchargées, calcul des rendements ...")
    returns = np.log(data / data.shift(1)).dropna()

    # Créer dossier si besoin
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    returns.to_csv(out_path)
    print(f"✅ Fichier sauvegardé : {out_path}")
    return returns

if __name__ == "__main__":
    df = download_and_process()
    df2 = download_and_process(start="2023-01-01",
                               end = "2024-01-01",
                               out_path="data/processed/returns_eval.csv")

