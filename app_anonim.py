# Talep Analiz ve Karar Destek Sistemi | Kurumsal ve anonim arayuz
# Calistirma: python -m streamlit run app_anonim.py
# Gerekenler: pip install streamlit pandas numpy openpyxl statsmodels plotly

import warnings
from io import BytesIO

import plotly.graph_objects as go
from html import escape
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt

st.set_page_config(
    page_title="Talep Analiz ve Karar Destek Sistemi",
    page_icon="📈", layout="wide", initial_sidebar_state="expanded"
)

# --------------------------------------------------------
# 1. VERI OKUMA VE KONTROL
# --------------------------------------------------------
def excel_oku(dosya):
    veri = pd.read_excel(dosya)
    veri.columns = [str(c).strip() if not isinstance(c, pd.Timestamp) else c
                    for c in veri.columns]
    if "Malzeme Kodu" not in veri.columns:
        raise ValueError("'Malzeme Kodu' sutunu bulunamadi.")

    if {"Ay", "Satis"}.issubset(veri.columns):
        df = veri[["Malzeme Kodu", "Ay", "Satis"]].copy()
    else:
        aylar = []
        for c in veri.columns:
            if c == "Malzeme Kodu" or isinstance(c, (int, float)):
                continue
            if pd.notna(pd.to_datetime(c, errors="coerce")):
                aylar.append(c)
        if not aylar:
            raise ValueError("Ay sutunlari bulunamadi. YYYY-MM veya tarih basligi kullanin.")
        df = veri.melt(id_vars="Malzeme Kodu", value_vars=aylar,
                       var_name="Ay", value_name="Satis")

    if df["Malzeme Kodu"].isna().any():
        raise ValueError("Bos malzeme kodu var.")
    df["Malzeme Kodu"] = df["Malzeme Kodu"].astype(str).str.strip()
    df["Ay"] = pd.to_datetime(df["Ay"], errors="coerce")
    df["Satis"] = pd.to_numeric(df["Satis"], errors="coerce")
    if df[["Ay", "Satis"]].isna().any().any():
        raise ValueError("Eksik/gecersiz ay veya satis var. Eksik satis sifir olarak varsayilmaz.")
    if (df["Satis"] < 0).any() or not np.isfinite(df["Satis"]).all():
        raise ValueError("Negatif veya sonlu olmayan satis miktari var.")
    if (df["Malzeme Kodu"] == "").any():
        raise ValueError("Bos malzeme kodu var.")
    df["Ay"] = df["Ay"].dt.to_period("M").dt.to_timestamp()
    if df.duplicated(["Malzeme Kodu", "Ay"]).any():
        raise ValueError("Tekrarlanan malzeme-ay kaydi var.")
    df = df.sort_values(["Malzeme Kodu", "Ay"]).reset_index(drop=True)
    for urun, grup in df.groupby("Malzeme Kodu"):
        beklenen = pd.date_range(grup["Ay"].min(), grup["Ay"].max(), freq="MS")
        if len(grup) != len(beklenen):
            raise ValueError(f"{urun}: Eksik ay var. Kaydi kontrol edin.")
    return df


def talep_sinifi(y):
    y = np.asarray(y, dtype=float)
    poz = y[y > 0]
    if len(poz) == 0:
        return np.nan, np.nan, "Talep Yok"
    adi = len(y) / len(poz)
    if len(poz) == 1:
        return adi, np.nan, "Yetersiz Veri"
    cv2 = float((np.std(poz, ddof=1) / np.mean(poz)) ** 2)
    if adi < 1.32:
        sinif = "Smooth" if cv2 < 0.49 else "Erratic"
    else:
        sinif = "Intermittent" if cv2 < 0.49 else "Lumpy"
    return adi, cv2, sinif


# --------------------------------------------------------
# 2. TAHMIN MODELLERI
# --------------------------------------------------------
def ses(y, alpha, h):
    duzey = float(y[0])
    for x in y[1:]:
        duzey = alpha * x + (1 - alpha) * duzey
    return np.full(h, max(0.0, duzey))


def croston(y, alpha, h, sba=False):
    indeks = np.flatnonzero(y > 0)
    if len(indeks) == 0:
        return np.zeros(h)
    z = float(y[indeks[0]])
    aralik = float(indeks[0] + 1)
    son = indeks[0]
    for t in indeks[1:]:
        z = alpha * y[t] + (1 - alpha) * z
        aralik = alpha * (t - son) + (1 - alpha) * aralik
        son = t
    sonuc = z / aralik
    if sba:
        sonuc *= 1 - alpha / 2
    return np.full(h, max(0.0, sonuc))


def tsb(y, alpha, beta, h):
    indeks = np.flatnonzero(y > 0)
    if len(indeks) == 0:
        return np.zeros(h)
    ilk = indeks[0]
    z = float(y[ilk])
    p = 1 / (ilk + 1)
    for t in range(ilk + 1, len(y)):
        p = beta * float(y[t] > 0) + (1 - beta) * p
        if y[t] > 0:
            z = alpha * y[t] + (1 - alpha) * z
    return np.full(h, max(0.0, p * z))


def adida(y, h):
    # Tamamlanmis 3 aylik bloklari topla; blok tahminini aylik ortalamaya cevir.
    blok_sayisi = len(y) // 3
    bloklar = y[-3 * blok_sayisi:].reshape(blok_sayisi, 3).sum(axis=1)
    return ses(bloklar, 0.3, h) / 3


def temel_tahmin(y, kod, h):
    y = np.asarray(y, dtype=float)
    if h < 1 or len(y) < 2:
        raise ValueError("Yetersiz tarihsel veri veya tahmin ufku")
    if kod == "Naive":
        return np.full(h, y[-1])
    if kod == "Ortalama":
        return np.full(h, y.mean())
    if kod == "MA3":
        return np.full(h, y[-3:].mean())
    if kod == "MA12":
        return np.full(h, y[-12:].mean())
    if kod == "SNaive12":
        if len(y) < 12:
            raise ValueError("12 ay gereklidir")
        return np.resize(y[-12:], h).astype(float)
    if kod.startswith("SES_"):
        return ses(y, float(kod.split("_")[1]), h)
    if kod.startswith("Croston_"):
        return croston(y, float(kod.split("_")[1]), h)
    if kod.startswith("SBA_"):
        return croston(y, float(kod.split("_")[1]), h, sba=True)
    if kod.startswith("TSB_"):
        a, b = map(float, kod.split("_")[1:])
        return tsb(y, a, b, h)
    if kod == "ADIDA3":
        if len(y) < 12:
            raise ValueError("ADIDA icin en az 12 ay gerekli")
        return adida(y, h)
    if kod == "Holt":
        fitted = Holt(y, initialization_method="estimated").fit(optimized=True)
        return np.asarray(fitted.forecast(h), dtype=float)
    if kod == "DampedHolt":
        fitted = Holt(y, damped_trend=True, initialization_method="estimated").fit(
            optimized=True, damping_trend=0.90)
        return np.asarray(fitted.forecast(h), dtype=float)
    if kod == "ETS_Mevsim":
        if len(y) < 24:
            raise ValueError("Mevsimsel ETS icin en az 24 ay gerekir")
        fitted = ExponentialSmoothing(
            y, trend="add", seasonal="add", seasonal_periods=12,
            initialization_method="estimated").fit(optimized=True)
        return np.asarray(fitted.forecast(h), dtype=float)
    if kod == "Theta":
        fitted = ThetaModel(y, period=12, deseasonalize=False).fit()
        return np.asarray(fitted.forecast(h), dtype=float)
    if kod == "Hibrit_SBA_TSB":
        return (temel_tahmin(y, "SBA_0.1", h) +
                temel_tahmin(y, "TSB_0.1_0.1", h)) / 2
    if kod == "Hibrit_SES_Theta":
        return (temel_tahmin(y, "SES_0.3", h) +
                temel_tahmin(y, "Theta", h)) / 2
    raise ValueError(f"Bilinmeyen model: {kod}")


def adaylar():
    kodlar = ["Naive", "Ortalama", "MA3", "MA12", "SNaive12",
              "Holt", "DampedHolt", "Theta", "ETS_Mevsim", "ADIDA3"]
    for a in [0.1, 0.3, 0.6]:
        kodlar.append(f"SES_{a}")
    for a in [0.05, 0.1, 0.3]:
        kodlar.extend([f"Croston_{a}", f"SBA_{a}"])
    for a in [0.05, 0.1, 0.3]:
        for b in [0.05, 0.1, 0.3]:
            kodlar.append(f"TSB_{a}_{b}")
    kodlar += ["Hibrit_SBA_TSB", "Hibrit_SES_Theta"]
    return kodlar


# --------------------------------------------------------
# 3. HATA METRIKLERI VE SINIFA GORE SECIM PUANI
# --------------------------------------------------------
def metrikler(gercek, tahmin, egitim):
    y = np.asarray(gercek, dtype=float)
    f = np.asarray(tahmin, dtype=float)
    hata = y - f
    ae = np.abs(hata)
    fark = np.diff(np.asarray(egitim, dtype=float))
    mas_p = float(np.mean(np.abs(fark))) if len(fark) else 0.0
    rms_p = float(np.mean(fark ** 2)) if len(fark) else 0.0
    pozitif = y > 0
    # y=0 ve f=0 icin MAAPE acisi 0; y=0 ve f>0 icin pi/2 limiti.
    aci = np.zeros(len(y), dtype=float)
    aci[pozitif] = np.arctan(ae[pozitif] / y[pozitif])
    aci[~pozitif & (f > 1e-10)] = np.pi / 2
    return {
        "MAE": float(ae.mean()),
        "RMSE": float(np.sqrt(np.mean(hata ** 2))),
        "WAPE_%": float(100 * ae.sum() / y.sum()) if y.sum() > 0 else np.nan,
        "MASE": float(ae.mean() / mas_p) if mas_p > 1e-10 else np.nan,
        "RMSSE": float(np.sqrt(np.mean(hata ** 2) / rms_p)) if rms_p > 1e-10 else np.nan,
        "MAAPE_rad": float(aci.mean()),
        "ME": float(hata.mean()),
        "Toplam_Hata": float(hata.sum()),
        "Toplam_Hata_%": float(100 * hata.sum() / y.sum()) if y.sum() > 0 else np.nan,
        "Pozitif_Ay_MAE": float(ae[pozitif].mean()) if pozitif.any() else np.nan,
        "Eksik_Tahmin_Ort": float(np.maximum(hata, 0).mean()),
        "Sifir_Tahmin_%": float(100 * np.mean(f < 0.5)),
    }


AGIRLIKLAR = {
    "Smooth": (0.55, 0.30, 0.15, 0.00),
    "Erratic": (0.30, 0.50, 0.00, 0.20),
    "Intermittent": (0.35, 0.10, 0.30, 0.25),
    "Lumpy": (0.30, 0.35, 0.20, 0.15),
    "Talep Yok": (0.50, 0.20, 0.20, 0.10),
    "Yetersiz Veri": (0.40, 0.30, 0.15, 0.15),
}


def secim_puani(m, sinif, ortalama_egitim):
    w_mae, w_rmse, w_bias, w_eksik = AGIRLIKLAR[sinif]
    # Olcek tum adaylar icin ayni. Puan kuculdukce daha iyi.
    pay = (w_mae * m["MAE"] + w_rmse * m["RMSE"] +
           w_bias * abs(m["ME"]) + w_eksik * m["Eksik_Tahmin_Ort"])
    return pay / max(1.0, ortalama_egitim)


def dogrulama_katlari(n):
    # Sadece egitim kesitinin ICINDE Rolling-Origin.
    h = min(12, max(6, n // 3))
    ilk = max(12, n // 2)
    son = n - h
    if ilk >= son:
        ilk = max(12, son - 2)
    katlar = sorted(set(np.linspace(ilk, son, 3, dtype=int).tolist()))
    if len(katlar) < 2:
        raise ValueError("Rolling-Origin icin yeterli egitim zamani yok.")
    return [(origin, h) for origin in katlar]


def model_ailesi(kod):
    if kod.startswith("TSB_"):
        return "TSB"
    if kod.startswith("SBA_"):
        return "SBA"
    if kod.startswith("Croston_"):
        return "Croston"
    if kod.startswith("SES_"):
        return "SES"
    return kod


def model_sec_ve_tahmin(y, ufuk):
    # Buraya yalnizca kullanima izin verilen EGITIM bolumu gelir.
    y = np.asarray(y, dtype=float)
    if len(y) < 30:
        raise ValueError("Model secimi icin en az 30 aylik egitim verisi gereklidir.")

    adi, cv2, sinif = talep_sinifi(y)
    katlar = dogrulama_katlari(len(y))
    cv_satirlari = []
    cv_tahminleri = {}
    atlananlar = []

    for kod in adaylar():
        gercekler, tahminler = [], []
        hata = None
        for origin, h in katlar:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tah = np.asarray(temel_tahmin(y[:origin], kod, h), dtype=float)
                if len(tah) != h or not np.isfinite(tah).all():
                    raise ValueError("Sonlu tahmin olusturulamadi")
                tah = np.maximum(0.0, tah)
                gercekler.extend(y[origin:origin + h])
                tahminler.extend(tah)
            except Exception as e:
                hata = str(e)
                break
        if hata is not None:
            atlananlar.append({"Model": kod, "Neden": hata})
            continue
        m = metrikler(gercekler, tahminler, y[:katlar[0][0]])
        puan = secim_puani(m, sinif, float(y.mean()))
        cv_satirlari.append({"Model": kod, "Aile": model_ailesi(kod),
                            "Secim_Puani": puan, **m})
        cv_tahminleri[kod] = (np.asarray(gercekler), np.asarray(tahminler))

    if not cv_satirlari:
        raise ValueError("Hicbir model dogrulanamadi.")
    tablo = pd.DataFrame(cv_satirlari).sort_values(
        ["Secim_Puani", "MAE", "RMSE"]).reset_index(drop=True)

    # Her aileden en iyi adayi al, iki farkli aileyi birlestir.
    tekiller = tablo[~tablo["Model"].str.startswith("Hibrit_")]
    farkli = tekiller.drop_duplicates("Aile")
    ilk_iki = farkli.head(2)
    eniyi_iki = ilk_iki["Model"].tolist()
    agirliklar = None

    # Ilk iki farkli ailenin basit ve ters-MAE agirlikli kombinasyonunu
    # gorunur bir aday olarak karsilastiriyoruz. Agirliklar ayni CV'den
    # ogrenildigi icin bu iki satirin CV puanlari iyimser olabilir.
    if len(eniyi_iki) == 2:
        k1, k2 = eniyi_iki
        ger = cv_tahminleri[k1][0]
        t1 = cv_tahminleri[k1][1]
        t2 = cv_tahminleri[k2][1]
        e1 = float(ilk_iki.iloc[0]["MAE"])
        e2 = float(ilk_iki.iloc[1]["MAE"])
        ters = np.array([1 / (e1 + 1e-6), 1 / (e2 + 1e-6)])
        agirliklar = ters / ters.sum()
        for kod, a, b in [
            ("Hibrit_Top2_Esit", 0.5, 0.5),
            ("Hibrit_Top2_Agirlikli", float(agirliklar[0]), float(agirliklar[1]))
        ]:
            mt = metrikler(ger, a * t1 + b * t2, y[:katlar[0][0]])
            tablo = pd.concat([tablo, pd.DataFrame([{
                "Model": kod, "Aile": "Hibrit",
                "Secim_Puani": secim_puani(mt, sinif, float(y.mean())), **mt
            }])], ignore_index=True)

    tablo = tablo.sort_values(["Secim_Puani", "MAE", "RMSE"]).reset_index(drop=True)
    kazanan = str(tablo.iloc[0]["Model"])

    def uret(kod):
        if kod == "Hibrit_Top2_Esit":
            return 0.5 * temel_tahmin(y, eniyi_iki[0], ufuk) + 0.5 * temel_tahmin(y, eniyi_iki[1], ufuk)
        if kod == "Hibrit_Top2_Agirlikli":
            return (agirliklar[0] * temel_tahmin(y, eniyi_iki[0], ufuk) +
                    agirliklar[1] * temel_tahmin(y, eniyi_iki[1], ufuk))
        return temel_tahmin(y, kod, ufuk)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        secilen_tahmin = np.maximum(0.0, np.asarray(uret(kazanan), dtype=float))
    if not np.isfinite(secilen_tahmin).all():
        raise ValueError("Secilen model sonlu tahmin uretmedi.")

    ayrinti = {
        "Sinif": sinif, "ADI": adi, "CV2": cv2, "Model": kazanan,
        "Puan": float(tablo.iloc[0]["Secim_Puani"]),
        "Kat_Sayisi": len(katlar), "CV_Ufuk": katlar[0][1],
        "Egitim_Ayi": len(y), "EnIyi_Iki": eniyi_iki,
        "Agirliklar": agirliklar, "Atlananlar": pd.DataFrame(atlananlar),
    }
    return secilen_tahmin, tablo, ayrinti


# Tekrar secilen malzemelerde ayni hesaplamayi yeniden yapmayalim.
model_sec_ve_tahmin = st.cache_data(show_spinner=False)(model_sec_ve_tahmin)

# --------------------------------------------------------
# 4. TRACKING SIGNAL VE ANOMALI MOTORU
# --------------------------------------------------------
def secilen_modelin_tahmini(y, detay, ufuk):
    """Model secimini DEGISTIRMEDEN, secilen modeli yeni tarihe uyarlar."""
    kod = str(detay["Model"])
    if kod in ("Hibrit_Top2_Esit", "Hibrit_Top2_Agirlikli"):
        m1, m2 = detay["EnIyi_Iki"]
        if kod == "Hibrit_Top2_Esit":
            a, b = 0.5, 0.5
        else:
            a, b = map(float, detay["Agirliklar"])
        t = a * temel_tahmin(y, m1, ufuk) + b * temel_tahmin(y, m2, ufuk)
    else:
        t = temel_tahmin(y, kod, ufuk)
    t = np.maximum(0.0, np.asarray(t, dtype=float))
    if not np.isfinite(t).all() or len(t) != ufuk:
        raise ValueError("Tracking Signal icin tahmin sayisi/sonlulugu hatali.")
    return t


def tracking_signal_hesapla(egitim, gercek, aylar, detay, limit=4.0):
    """Dis ornek one-step walk-forward izleme.

    Model ilk egitimde secilir; test gercegi gozlendikce ayni model
    ve SABIT parametrelerle yeniden egitilir. Hicbir test ayinin
    gercegi o ayin tahminini olusturmakta kullanilmaz.
    """
    gecmis = np.asarray(egitim, float).copy()
    gercek = np.asarray(gercek, float)
    aylar = pd.to_datetime(aylar)
    if len(gercek) != len(aylar):
        raise ValueError("Tracking Signal tarih ve satis uzunluklari ayni olmali.")
    rsfe = 0.0
    mutlak_toplam = 0.0
    satirlar = []
    for i, (ay, g) in enumerate(zip(aylar, gercek), start=1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = float(secilen_modelin_tahmini(gecmis, detay, 1)[0])
        hata = float(g - t)
        rsfe += hata
        mutlak_toplam += abs(hata)
        mad = mutlak_toplam / i
        ts = rsfe / mad if mad > 1e-10 else 0.0
        if ts > limit:
            durum = "Eksik tahmin uyarisi"
        elif ts < -limit:
            durum = "Fazla tahmin uyarisi"
        else:
            durum = "Kontrol icinde"
        satirlar.append({
            "Ay": pd.Timestamp(ay), "Gercek_Satis": float(g),
            "Bir_Adim_Tahmin": t, "Hata": hata,
            "RSFE": rsfe, "Kumulatif_MAD": mad, "TS": ts,
            "Durum": durum,
        })
        gecmis = np.append(gecmis, g)
    return pd.DataFrame(satirlar)


def anomali_tespit(df, bakis=24, asgari_gecmis=12,
                   asgari_pozitif=6, sifir_esik=0.05):
    """Tek yonlu tarihsel analiz: karar yalnizca onceki aylara dayanir.

    Yuksek satis: onceki en cok 24 aydaki pozitif satislarin
    Q3+1.5*IQR veya 1.3*Q3 esiginin asimi.
    Beklenmedik sifir seri: onceki en cok 24 ayin duzeltilmis
    talep olasiligi p=(k+1)/(n+2); (1-p)^L<0.05, L>=2.
    Bir sifir kosusunda sadece ilk esik asiminda alarm verir.
    """
    satirlar = []
    for kod, grup in df.groupby("Malzeme Kodu"):
        grup = grup.sort_values("Ay")
        y = grup["Satis"].to_numpy(float)
        aylar = pd.to_datetime(grup["Ay"]).to_numpy()
        sifir_uzunlugu = 0
        sifir_alarm_verildi = False
        for i, v in enumerate(y):
            onceki = y[max(0, i - bakis):i]
            poz = onceki[onceki > 0]
            ay = pd.Timestamp(aylar[i])
            if i >= asgari_gecmis and len(poz) >= asgari_pozitif and v > 0:
                q1, q3 = np.percentile(poz, [25, 75])
                esik = max(float(q3 + 1.5 * (q3 - q1)), float(1.3 * q3))
                if float(v) > esik:
                    satirlar.append({
                        "Malzeme Kodu": kod, "Ay": ay,
                        "Anomali Turu": "Yuksek satis sicramasi",
                        "Gercek Satis": float(v), "Esik": esik,
                        "Olasilik": np.nan, "Sifir Seri Uzunlugu": 0,
                        "Aciklama": "Gecmis pozitif satislarin ust IQR esigi asildi",
                    })
            if v == 0:
                sifir_uzunlugu += 1
                if (not sifir_alarm_verildi and sifir_uzunlugu >= 2
                        and i >= asgari_gecmis and len(poz) >= 3):
                    p = (len(poz) + 1) / (len(onceki) + 2)
                    olasilik = (1 - p) ** sifir_uzunlugu
                    if olasilik < sifir_esik:
                        satirlar.append({
                            "Malzeme Kodu": kod, "Ay": ay,
                            "Anomali Turu": "Beklenmedik sifir satis serisi",
                            "Gercek Satis": 0.0, "Esik": np.nan,
                            "Olasilik": float(olasilik),
                            "Sifir Seri Uzunlugu": int(sifir_uzunlugu),
                            "Aciklama": "(1-p)^L<0.05; bu seri icin ilk uyari",
                        })
                        sifir_alarm_verildi = True
            else:
                sifir_uzunlugu = 0
                sifir_alarm_verildi = False
    sutunlar = ["Malzeme Kodu", "Ay", "Anomali Turu", "Gercek Satis",
               "Esik", "Olasilik", "Sifir Seri Uzunlugu", "Aciklama"]
    return pd.DataFrame(satirlar, columns=sutunlar).sort_values(
        ["Malzeme Kodu", "Ay"]
    ).reset_index(drop=True) if satirlar else pd.DataFrame(columns=sutunlar)


def rapor_excel_uret(veriler):
    tampon = BytesIO()
    with pd.ExcelWriter(tampon, engine="openpyxl", datetime_format="YYYY-MM-DD") as writer:
        for ad, veri in veriler.items():
            if not isinstance(veri, pd.DataFrame):
                continue
            guvenli_ad = ad[:31]
            veri.to_excel(writer, sheet_name=guvenli_ad, index=False)
            ws = writer.sheets[guvenli_ad]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            from openpyxl.styles import Font, PatternFill, Alignment
            for hucre in ws[1]:
                hucre.font = Font(bold=True, color="FFFFFF")
                hucre.fill = PatternFill("solid", fgColor="17365D")
                hucre.alignment = Alignment(wrap_text=True)
            for sutun in ws.columns:
                baslik = str(sutun[0].value or "")
                genislik = min(34, max(13, len(baslik) + 3))
                ws.column_dimensions[sutun[0].column_letter].width = genislik
    return tampon.getvalue()


# --------------------------------------------------------
# 5. TOPLU RAPOR HESAPLAMA (GELECEK + GECMIS TEST)
# --------------------------------------------------------
def toplu_analiz(df, egitim_ay, test_tamami=False, ilerleme=None):
    """Her Excel yuklemesinde yalnizca yuklenen satislar kullanilir."""
    secim_rows, test_rows, gelecek_rows, cv_rows, ts_rows, hatalar = [], [], [], [], [], []
    sirali = list(df.groupby("Malzeme Kodu"))
    anomaliler = anomali_tespit(df)
    anomali_sayim = (anomaliler.groupby("Malzeme Kodu").size().to_dict()
                      if not anomaliler.empty else {})
    for i, (kod, grup) in enumerate(sirali):
        grup = grup.sort_values("Ay")
        y = grup["Satis"].to_numpy(float)
        n = len(y)
        try:
            if n <= egitim_ay or egitim_ay < 30:
                raise ValueError(f"{n} ay veri var; egitim {egitim_ay} ay olamaz.")
            train = y[:egitim_ay]
            test_h = n - egitim_ay if test_tamami else min(12, n - egitim_ay)
            gercek = y[egitim_ay:egitim_ay + test_h]
            aylar_test = grup["Ay"].iloc[egitim_ay:egitim_ay + test_h].reset_index(drop=True)
            pred_test, cv_test, detay_test = model_sec_ve_tahmin(train, test_h)
            performans = metrikler(gercek, pred_test, train)

            # Ayri deney: tum tarihsel veriyle gercek gelecek 12 ayi.
            gelecek, cv_gelecek, detay_gelecek = model_sec_ve_tahmin(y, 12)
            son_ay = pd.Timestamp(grup["Ay"].iloc[-1])
            gelecek_aylar = pd.date_range(son_ay + pd.offsets.MonthBegin(1), periods=12, freq="MS")

            ts = tracking_signal_hesapla(train, gercek, aylar_test, detay_test)
            ts_son = float(ts["TS"].iloc[-1]) if not ts.empty else np.nan
            ts_ihlal = int((ts["TS"].abs() > 4).sum()) if not ts.empty else 0
            sinif_adi = detay_test["Sinif"]
            secim_rows.append({
                "Malzeme Kodu": kod, "Talep Sinifi (egitim)": sinif_adi,
                "ADI (egitim)": detay_test["ADI"], "CV2 (egitim)": detay_test["CV2"],
                "Test_Model": detay_test["Model"], "Test_CV_Puani": detay_test["Puan"],
                "Gelecek_Model": detay_gelecek["Model"],
                "Gelecek_CV_Puani": detay_gelecek["Puan"],
                "Test_MAE": performans["MAE"], "Test_RMSE": performans["RMSE"],
                "Test_WAPE_%": performans["WAPE_%"], "Test_MASE": performans["MASE"],
                "Test_RMSSE": performans["RMSSE"], "Test_ME": performans["ME"],
                "Test_Toplam_Hata": performans["Toplam_Hata"],
                "TS_Son": ts_son, "TS_Uyari_Ay": ts_ihlal,
                "Anomali_Sayisi": int(anomali_sayim.get(kod, 0)),
            })
            for ay, t, g in zip(aylar_test, pred_test, gercek):
                test_rows.append({"Malzeme Kodu": kod, "Ay": ay,
                                  "Model": detay_test["Model"],
                                  "Gercek Satis": float(g), "Tahmin": float(t),
                                  "Hata": float(g - t)})
            for ay, t in zip(gelecek_aylar, gelecek):
                gelecek_rows.append({"Malzeme Kodu": kod, "Ay": ay,
                                     "Model": detay_gelecek["Model"],
                                     "Tahmin": float(t)})
            for evre, tablo in [("Gecmis test egitimi", cv_test), ("Tum tarihsel veri", cv_gelecek)]:
                for _, row in tablo.iterrows():
                    cv_rows.append({"Malzeme Kodu": kod, "Evre": evre, **row.to_dict()})
            ts["Malzeme Kodu"] = kod
            ts_rows.extend(ts.to_dict("records"))
        except Exception as e:
            hatalar.append({"Malzeme Kodu": kod, "Hata": f"{type(e).__name__}: {e}"})
        if ilerleme is not None:
            ilerleme((i + 1) / len(sirali))

    siniflar = []
    for kod, g in sirali:
        adi, cv2, sinif = talep_sinifi(g.sort_values("Ay")["Satis"].to_numpy(float))
        siniflar.append({"Malzeme Kodu": kod, "ADI": adi, "CV2": cv2,
                         "Talep Sinifi": sinif})
    metodoloji = pd.DataFrame([
        {"Konu": "Veri", "Aciklama": "Yuklenen Excel; sifir satis eksik veri degildir. Ornek sistem testleri sentetik verilerle gerceklestirilmistir."},
        {"Konu": "Model secimi", "Aciklama": "Her urun icin Rolling-Origin; egitim disindaki gercekler model secimine girmez."},
        {"Konu": "CV secim puani", "Aciklama": "Talep sinifina gore deneysel agirlikli MAE, RMSE, |ME| ve eksik tahmin; olcek egitim ortalamasi."},
        {"Konu": "Gecmis test", "Aciklama": f"Ilk {egitim_ay} ay egitim; sonraki {'tum' if test_tamami else 'en fazla 12'} ay sabit ufuklu dis test."},
        {"Konu": "Tracking Signal", "Aciklama": "Secilen model bir adim ileri yurutulur ve gercek geldikce yeniden egitilir. TS=RSFE/kumulatif MAD; +/-4 izlemede kullanilan deneysel esik."},
        {"Konu": "TS ve test farki", "Aciklama": "Tracking Signal tek adim ileri guncellenen tahminlere; test MAE sabit baslangictan cok adimli tahminlere dayanir."},
        {"Konu": "Sicrama anomalisi", "Aciklama": "Gecmis en cok 24 ayin 6+ pozitif satisindan Q3+1.5 IQR veya 1.3 Q3 esigi; en az 12 ay gecmis."},
        {"Konu": "Sifir seri anomalisi", "Aciklama": "p=(k+1)/(n+2), (1-p)^L<0.05 ve L>=2; yalnizca ilk alarm; istatistiksel on-inceleme uyarisi."},
        {"Konu": "Veri sizintisi", "Aciklama": "Anomali esikleri yalnizca ilgili aydan ONCEKI gozlemlerden hesaplanir."},
        {"Konu": "Metrik siniri", "Aciklama": "WAPE gercek toplam sifirken, MASE/RMSSE egitim fark olcegi sifirken tanimsizdir."},
        {"Konu": "Kisit", "Aciklama": "Stok seviyesi, teslim suresi ve dogrulanmis anomali etiketi olmadan stok faydasi ve anomali tespit dogrulugu olculemez."},
    ])
    return {
        "Ozet": pd.DataFrame(secim_rows),
        "Gelecek_12_Ay": pd.DataFrame(gelecek_rows),
        "Gecmis_Test_Aylik": pd.DataFrame(test_rows),
        "Tracking_Signal": pd.DataFrame(ts_rows),
        "Anomali_Detay": anomaliler,
        "Talep_Siniflari": pd.DataFrame(siniflar),
        "Model_CV_Tum": pd.DataFrame(cv_rows),
        "Metodoloji": metodoloji,
        "Hatalar": pd.DataFrame(hatalar, columns=["Malzeme Kodu", "Hata"]),
    }


# --------------------------------------------------------
# 6. KOMPAKT VE ETKILESIMLI KULLANICI ARAYUZU
# --------------------------------------------------------
PALET = {
    # Koyu, yuksek kontrastli zemin + yumusak pastel vurgu renkleri
    "ink": "#E9EFF8", "muted": "#A5B5CC", "line": "#324255",
    "actual": "#93B9E4", "forecast": "#82D3BB", "positive": "#82D3BB",
    "negative": "#F1A0A6", "warning": "#F3C384", "neutral": "#A6B6CC",
    "canvas": "#152236", "violet": "#C8B4EC",
}

st.markdown("""
<style>
/* Koyu arka plan, pastel vurgu ve yuksek kontrastli veri gorunumu. */
:root { color-scheme: dark; }
html, body, .stApp {font-family: 'Segoe UI', system-ui, sans-serif !important;}
.stApp, [data-testid="stAppViewContainer"] {
    background: #0F1725 !important;
    color: #E9EFF8 !important;
}
[data-testid="stHeader"] {background: #0F1725 !important;}
[data-testid="stToolbar"] {color: #DDE8F7 !important;}
.block-container {padding-top: 1.15rem !important; padding-bottom: 1.8rem !important;
                  max-width: 1600px !important;}
h1, h2, h3, [data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {color: #E9EFF8 !important;}
h1 {font-size: 1.5rem !important; font-weight: 720 !important;}
h2 {font-size: 1.2rem !important; font-weight: 650 !important;}
h3 {font-size: 1.02rem !important; font-weight: 620 !important;}
p, label, [data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] p {font-size: .90rem; color: #DFE7F2 !important;}
[data-testid="stCaptionContainer"] p, .stCaption {color: #AABAD0 !important;}
[data-testid="stMetricValue"] {font-size: 1.15rem !important; font-weight: 700 !important; color:#E9EFF8 !important;}
[data-testid="stMetricLabel"] {font-size: .81rem !important; color: #B1C0D3 !important;}
[data-testid="stMetricDelta"] {font-size: .77rem !important;}
[data-testid="stSidebar"] {background: #151F30 !important; border-right: 1px solid #2B3B4F !important;
  min-width: 268px !important; max-width: 302px !important;}
[data-testid="stSidebar"] > div:first-child {width: 290px !important;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {color: #E9EFF8 !important;}
[data-testid="stSidebar"] .block-container {padding-top: 1.0rem !important;}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section,
[data-testid="stSidebar"] [data-baseweb="input"] {
    background: #1D2A40 !important;
    border-color: #3A4E66 !important;
    color: #E9EFF8 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: #30455F !important; color: #E9EFF8 !important; border-color:#4B627E !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #E9EFF8 !important;
}
[data-testid="stVerticalBlock"] {gap: .70rem !important;}
[data-testid="stTabs"] button {font-size: .92rem !important; font-weight: 600 !important;
  padding: .60rem .90rem !important; color:#BBCBDD !important;}
[data-testid="stTabs"] button[aria-selected="true"] {color: #C9B7FF !important;
 border-bottom-color: #B6A5E6 !important;}
[data-testid="stExpander"] {border-radius: 10px !important;
  background: #172335 !important; border:1px solid #34465F !important;}
[data-testid="stExpander"] summary {font-size: .90rem !important; color: #E1EAF8 !important;}
[data-testid="stAlert"] {padding: .65rem .85rem !important; border-radius: 10px !important;}
[data-testid="stDataFrame"] {border:1px solid #34465F !important; border-radius:10px; overflow:hidden;}
.stPlotlyChart, [data-testid="stPlotlyChart"] {
    border: 1px solid #354760; border-radius: 12px;
    padding: .15rem; background: #152236 !important;
    animation: chart-in .30s ease-out both;
}
@keyframes chart-in {from {opacity: 0; transform: translateY(3px)}
                    to {opacity: 1; transform: translateY(0)}}
.panel-title {font-size: 1.02rem; font-weight: 680; margin: .55rem 0 .45rem; color:#E5EDFA;}
.dash-header {display:flex; gap:.6rem; align-items:center; justify-content:space-between;
    border-bottom:1px solid #2B3B50; padding:.3rem 0 .75rem; margin-bottom:.65rem;}
.dash-brand {font-size:1.39rem; font-weight:740; letter-spacing:-.02em; color:#EEF3FD;}
.dash-sub {font-size:.82rem; color:#B1C0D3;}
.chips {display:flex; flex-wrap:wrap; gap:.48rem; margin:.3rem 0 .62rem;}
.chip {display:inline-flex; align-items:center; gap:.30rem;
       background:#1B293C; border:1px solid #394B61;
       border-radius:9px; padding:.42rem .72rem; font-size:.83rem;
       color:#B9C9DA; white-space:nowrap;}
.chip b {color:#F2F6FC; font-size:.94rem; margin-left:.13rem;}
.chip.positive {background:#19352F; border-color:#3D7A6C; color:#B2EAD7;}
.chip.positive b {color:#B8F4E0;}
.chip.negative {background:#422B37; border-color:#885569; color:#F6B8C2;}
.chip.negative b {color:#FFD0D6;}
.chip.warning {background:#453626; border-color:#897049; color:#F5D8AA;}
.chip.warning b {color:#FFE3B1;}
.chip.neutral {background:#1D293C; border-color:#40516A; color:#BDD0E6;}
.note {font-size:.85rem; color:#AAB9CE; margin:.15rem 0 .65rem;}
.callout {border-left:3px solid #B3C4E3; background:#223149;
          padding:.60rem .85rem; border-radius:8px; font-size:.86rem; color:#D7E4F3;}
.callout.negative {border-color:#F1A0A6; background:#412935; color:#FFD1D6;}
.callout.positive {border-color:#82D3BB; background:#18382F; color:#C6F6E5;}
.callout.warning {border-color:#F3C384; background:#493724; color:#FFE5B7;}
</style>
""", unsafe_allow_html=True)


def tr_num(v, n=0):
    if v is None or not np.isfinite(v):
        return "—"
    x = f"{float(v):,.{n}f}"
    return x.replace(",", "@").replace(".", ",").replace("@", ".")


def chip(label, value, status=""):
    return (f'<span class="chip {status}">{escape(str(label))}'
            f'<b>{escape(str(value))}</b></span>')


def chips_goster(items):
    st.markdown('<div class="chips">' + ''.join(chip(*item) for item in items) + '</div>',
                unsafe_allow_html=True)


def mesaj(metin, tip=""):
    st.markdown('<div class="callout ' + tip + '">' + escape(str(metin)) + '</div>',
                unsafe_allow_html=True)


def tema(fig, height=440, yatay_baslik=None):
    fig.update_layout(
        autosize=True,
        height=height,
        paper_bgcolor=PALET["canvas"],
        plot_bgcolor=PALET["canvas"],
        font=dict(family="Segoe UI, Arial, sans-serif", color=PALET["ink"], size=14),
        margin=dict(l=58, r=28, t=80, b=58),
        showlegend=True,
        legend=dict(orientation="h", y=1.06, x=0, yanchor="bottom",
                    font=dict(size=14, color=PALET["ink"]),
                    bgcolor="rgba(0,0,0,0)", itemsizing="constant"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1D2C44", font=dict(color="#F0F6FF", size=13)),
        transition=dict(duration=260, easing="cubic-in-out"),
        xaxis=dict(showgrid=False, showline=True, linecolor=PALET["line"],
                   title=dict(text=yatay_baslik, font=dict(size=14, color=PALET["ink"])),
                   tickformat="%m/%Y", tickfont=dict(size=13, color="#C8D6E8")),
        yaxis=dict(gridcolor=PALET["line"], zeroline=False,
                   title=dict(text="Adet" if yatay_baslik is None else None,
                              font=dict(size=14, color=PALET["ink"])),
                   tickfont=dict(size=13, color="#C8D6E8")),
    )
    return fig


GRAFIK_KONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["toImage", "lasso2d", "select2d"],
    "scrollZoom": True,
    "responsive": True,
}


def ciz(fig, key):
    st.plotly_chart(fig, use_container_width=True, config=GRAFIK_KONFIG, key=key)


def tahmin_grafigi(grup, egitim_ay, sonuc, gecmis_mi):
    fig = go.Figure()
    if gecmis_mi:
        train = grup.iloc[:egitim_ay]
        test = grup.iloc[egitim_ay:egitim_ay+len(sonuc)]
        fig.add_trace(go.Scatter(
            x=train["Ay"], y=train["Satis"], mode="lines+markers",
            name="Eğitim satışları", line=dict(color=PALET["actual"], width=2.6),
            marker=dict(size=4), hovertemplate="%{x|%Y-%m}<br>Eğitim: %{y:,.0f} adet<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=test["Ay"], y=test["Satis"], mode="lines+markers",
            name="Test gerçekleri", line=dict(color=PALET["violet"], width=2.8),
            marker=dict(size=5), hovertemplate="%{x|%Y-%m}<br>Gerçek: %{y:,.0f} adet<extra></extra>"))
        fig.add_vline(x=pd.Timestamp(grup["Ay"].iloc[egitim_ay]).timestamp()*1000,
                      line_dash="dot", line_color=PALET["muted"], opacity=.6)
    else:
        fig.add_trace(go.Scatter(
            x=grup["Ay"], y=grup["Satis"], mode="lines+markers",
            name="Geçmiş satış", line=dict(color=PALET["actual"], width=2.6),
            marker=dict(size=4), hovertemplate="%{x|%Y-%m}<br>Satış: %{y:,.0f} adet<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=sonuc["Ay"], y=sonuc["Tahmin"], mode="lines+markers",
        name="Model tahmini", line=dict(color=PALET["forecast"], width=3.2),
        marker=dict(size=6, symbol="circle", line=dict(color="white", width=1)),
        hovertemplate="%{x|%Y-%m}<br>Tahmin: %{y:,.1f} adet<extra></extra>"))
    tema(fig, 525)
    fig.update_layout(yaxis_title="Satış (adet)")
    return fig


def ts_grafigi(ts):
    fig = go.Figure()
    fig.add_hrect(y0=-4, y1=4, fillcolor="#EAF6EF", opacity=.65, line_width=0)
    fig.add_trace(go.Scatter(
        x=ts["Ay"], y=ts["TS"], name="Tracking Signal", mode="lines+markers",
        line=dict(color=PALET["actual"], width=2.6), marker=dict(size=5),
        customdata=ts[["RSFE", "Kumulatif_MAD"]].to_numpy(),
        hovertemplate="%{x|%Y-%m}<br>TS: %{y:+.2f}<br>RSFE: %{customdata[0]:,.1f}"
                      "<br>MAD: %{customdata[1]:,.1f}<extra></extra>"))
    for lim, isim, col in [(4, "+4 · Eksik tahmin", PALET["negative"]),
                           (-4, "−4 · Fazla tahmin", PALET["warning"])]:
        fig.add_hline(y=lim, line_dash="dash", line_color=col,
                      annotation_text=isim, annotation_position="top right",
                      annotation_font_color=col, annotation_font_size=10)
    for kosul, col, ad in [(ts["TS"] > 4, PALET["negative"], "Eksik tahmin uyarısı"),
                          (ts["TS"] < -4, PALET["warning"], "Fazla tahmin uyarısı")]:
        part=ts[kosul]
        if not part.empty:
            fig.add_trace(go.Scatter(
                x=part["Ay"], y=part["TS"], mode="markers", name=ad,
                marker=dict(size=10, color=col, line=dict(width=1, color="white")),
                hovertemplate="%{x|%Y-%m}<br>TS: %{y:+.2f}<extra></extra>"))
    tema(fig, 505)
    fig.update_layout(yaxis_title="TS = RSFE / MAD")
    return fig


def anomali_grafigi(grup, anom):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grup["Ay"], y=grup["Satis"], mode="lines+markers", name="Gerçek satış",
        line=dict(color=PALET["actual"], width=2.6), marker=dict(size=4),
        hovertemplate="%{x|%Y-%m}<br>Satış: %{y:,.0f} adet<extra></extra>"))
    for tur, ad, color, symbol in [
        ("Yuksek satis sicramasi", "Yüksek satış", PALET["negative"], "diamond"),
        ("Beklenmedik sifir satis serisi", "Sıfır satış serisi", PALET["warning"], "x"),
    ]:
        part = anom[anom["Anomali Turu"] == tur]
        if not part.empty:
            if tur == "Yuksek satis sicramasi":
                custom = part[["Esik"]].to_numpy()
                hover = ("%{x|%Y-%m}<br>Satış: %{y:,.0f} adet"
                         "<br>Aşım eşiği: %{customdata[0]:,.1f}<extra></extra>")
            else:
                custom = part[["Olasilik", "Sifir Seri Uzunlugu"]].to_numpy()
                hover = ("%{x|%Y-%m}<br>Satış: 0"
                         "<br>Sıfır seri olasılığı: %{customdata[0]:.3f}"
                         "<br>Ardışık sıfır: %{customdata[1]:.0f}<extra></extra>")
            fig.add_trace(go.Scatter(
                x=part["Ay"], y=part["Gercek Satis"], mode="markers", name=ad,
                marker=dict(size=11, color=color, symbol=symbol,
                            line=dict(width=1, color="#FFFFFF")),
                customdata=custom, hovertemplate=hover))
    tema(fig, 505)
    fig.update_layout(yaxis_title="Satış (adet)")
    return fig


# Secimleri kapatilabilen yan menude tutuyoruz.
st.sidebar.markdown("### Analiz Ayarları")
uploaded = st.sidebar.file_uploader("Satış verisi (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.markdown('<div class="dash-brand">Talep Analiz ve Karar Destek Sistemi</div>',
                unsafe_allow_html=True)
    st.info("Analize başlamak için Excel dosyasını yükleyin.")
    st.stop()
try:
    df = excel_oku(uploaded)
except Exception as e:
    st.error(f"Excel kontrol hatası: {e}")
    st.stop()

urunler = sorted(df["Malzeme Kodu"].unique())
secili = st.sidebar.selectbox("Malzeme", urunler)
gr = df[df["Malzeme Kodu"] == secili].sort_values("Ay").copy()
y_all = gr["Satis"].to_numpy(dtype=float)

mod = st.sidebar.radio("Analiz modu", ["Geçmiş dönem testi", "Gelecek 12 ay"], horizontal=False)
if mod == "Geçmiş dönem testi":
    if len(y_all) < 36:
        st.error("Geçmiş dönem testi için en az 36 ay gerekli.")
        st.stop()
    egitim_ay = st.sidebar.slider("Eğitimde kullanılacak ilk ay", 30, len(y_all)-6,
                                 value=min(45, len(y_all)-6))
    test_secim = st.sidebar.radio("Test ufku", ["İlk 12 ay", "Kalan tüm aylar"], horizontal=False)
    h = (len(y_all)-egitim_ay if test_secim == "Kalan tüm aylar"
         else min(12, len(y_all)-egitim_ay))
    y_train, y_test = y_all[:egitim_ay], y_all[egitim_ay:egitim_ay+h]
    ay_tahmin = gr["Ay"].iloc[egitim_ay:egitim_ay+h].reset_index(drop=True)
else:
    egitim_ay, h, y_train, y_test = len(y_all), 12, y_all, None
    ay_tahmin = pd.date_range(gr["Ay"].iloc[-1] + pd.offsets.MonthBegin(1), periods=12, freq="MS")

with st.sidebar.expander("Doğrulama yöntemi", expanded=False):
    st.caption("Model, eğitim dönemi içindeki Rolling-Origin doğrulamasıyla seçilir. Test verisi seçimde kullanılmaz. Talep sınıfına özgü puanlar uygulama için belirlenmiş deneysel ağırlıklara dayanır.")

with st.spinner("Model ve doğrulama sonuçları hazırlanıyor..."):
    try:
        tahmin, cv, detay = model_sec_ve_tahmin(y_train, h)
    except Exception as e:
        st.error(f"Modelleme hatası: {e}")
        st.stop()

# Sekmelerin birbirini tekrar hesaplamamasi icin hafif ozetler.
sonuc_df = pd.DataFrame({"Ay": pd.to_datetime(ay_tahmin), "Tahmin": tahmin})
olcum = None
if y_test is not None:
    sonuc_df["Gercek Satis"] = y_test
    sonuc_df["Hata (Gercek-Tahmin)"] = y_test - tahmin
    olcum = metrikler(y_test, tahmin, y_train)

anom = anomali_tespit(df)
secili_anom = anom[anom["Malzeme Kodu"] == secili].copy()
yuksek_sayi = int((secili_anom["Anomali Turu"] == "Yuksek satis sicramasi").sum())
sifir_sayi = int((secili_anom["Anomali Turu"] == "Beklenmedik sifir satis serisi").sum())

st.markdown(
    '<div class="dash-header"><div><div class="dash-brand">Talep Analiz ve Karar Destek Sistemi</div>'
    '<div class="dash-sub">Talep Tahmini · Performans İzleme · Anomali Analizi</div>'
    '</div></div>', unsafe_allow_html=True)

# Basit degisim gostergesi sadece gecmis SATIS farkini gosterir, tahmin dogrulugu degildir.
son_uc = float(y_all[-3:].mean())
onceki_uc = float(y_all[-6:-3].mean()) if len(y_all) >= 6 else np.nan
if np.isfinite(onceki_uc) and onceki_uc > 0:
    degisim = 100*(son_uc-onceki_uc)/onceki_uc
    durum_degis = "positive" if degisim > 0 else "negative" if degisim < 0 else "neutral"
    degisim_yazi = f"{degisim:+.1f}%"
else:
    durum_degis, degisim_yazi = "neutral", "—"

chips_goster([
    ("Malzeme", secili, ""),
    ("Talep sınıfı", detay["Sinif"], "neutral"),
    ("Seçilen model", detay["Model"], ""),
    ("Gözlem", str(len(y_all))+" ay", ""),
    ("Son 3 ay değişim", degisim_yazi, durum_degis),
    ("Anomali", str(len(secili_anom))+" uyarı", "negative" if yuksek_sayi else "warning" if sifir_sayi else "positive"),
])

sekme_tahmin, sekme_ts, sekme_anom, sekme_cv, sekme_toplu = st.tabs([
    "Tahmin ve Test", "Tracking Signal", "Anomali Analizi", "Model Karşılaştırma", "Raporlama"
])

with sekme_tahmin:
    st.markdown('<div class="panel-title">Gerçek satış ve model tahmini</div>', unsafe_allow_html=True)
    # Grafik artik sayfanin tum kullanilabilir genisligini kaplar.
    ciz(tahmin_grafigi(gr, egitim_ay, sonuc_df, y_test is not None),
        "tahmin_"+secili+str(egitim_ay)+mod)

    st.markdown('<div class="panel-title">Test performansı</div>', unsafe_allow_html=True)
    if olcum:
        chips_goster([
            ("MAE", tr_num(olcum["MAE"], 1), ""),
            ("RMSE", tr_num(olcum["RMSE"], 1), ""),
            ("WAPE", tr_num(olcum["WAPE_%"], 1)+"%", ""),
            ("MASE", tr_num(olcum["MASE"], 3), ""),
            ("ME", tr_num(olcum["ME"], 1),
             "negative" if olcum["ME"]>0 else "warning" if olcum["ME"]<0 else "positive"),
        ])
        if olcum["Toplam_Hata"] > 0:
            mesaj(f"Toplam {tr_num(olcum['Toplam_Hata'],0)} adet eksik tahmin. Pozitif hata: gerçek satış tahminden yüksek.", "negative")
        elif olcum["Toplam_Hata"] < 0:
            mesaj(f"Toplam {tr_num(abs(olcum['Toplam_Hata']),0)} adet fazla tahmin. Negatif hata: tahmin gerçekleşenden yüksek.", "warning")
        else:
            mesaj("Toplam tahmin yanlılığı sıfır.", "positive")
    else:
        mesaj("Gelecekteki satışlar henüz gerçekleşmediğinden bağımsız test hatası hesaplanamaz.")

    with st.expander("Model seçim yöntemi", expanded=False):
        chips_goster([
            ("Seçilen model", detay["Model"], ""),
            ("Doğrulama puanı", tr_num(detay["Puan"],3), ""),
            ("Kayan başlangıç", str(detay["Kat_Sayisi"]), ""),
            ("CV ufku", str(detay["CV_Ufuk"])+" ay", ""),
            ("Eğitim", str(egitim_ay)+" ay", ""),
        ])
        w = AGIRLIKLAR[detay["Sinif"]]
        st.write(f"{detay['Sinif']} için eğitim içi seçim puanı: "
                 f"{w[0]:.0%} MAE + {w[1]:.0%} RMSE + "
                 f"{w[2]:.0%} |ME| + {w[3]:.0%} eksik tahmin ortalaması; "
                 "eğitim ortalamasıyla normalize edilir. Düşük puan daha iyidir.")
        st.caption("Talep sınıfına özgü ağırlıklar deneysel seçim kuralıdır; evrensel akademik eşikler değildir.")
        if detay["Model"].startswith("Hibrit_Top2"):
            st.write("Hibrit bileşenleri: " + ", ".join(detay["EnIyi_Iki"]))
            st.caption("Bileşen seçiminde aynı CV kullanıldığı için hibrit CV puanı iyimser olabilir.")

    with st.expander("Aylık tahmin ve hata metrikleri", expanded=False):
        sol_t, sag_t = st.columns([1.5, 1], gap="small")
        with sol_t:
            st.dataframe(sonuc_df.round(3), hide_index=True, use_container_width=True, height=265)
        with sag_t:
            if olcum:
                st.dataframe(pd.DataFrame([olcum]).T.rename(columns={0:"Değer"}).round(3),
                             use_container_width=True, height=265)
            else:
                st.caption("Gelecek için test hata değerleri yoktur.")
    if olcum and olcum["Sifir_Tahmin_%"] > 50:
        mesaj("Tahminlerin yarısından fazlası sıfıra yakın; eksik talep açısından incelenmeli.", "warning")

with sekme_ts:
    st.markdown('<div class="panel-title">Bir adım ileri tahminle Tracking Signal</div>', unsafe_allow_html=True)
    try:
        if y_test is not None:
            ts_egitim, ts_gercek, ts_aylar, ts_detay = y_train, y_test, ay_tahmin, detay
            ts_aciklama = f"İlk {egitim_ay} ayla seçilen model, test ayları geldikçe bir adım ilerletilir."
        else:
            izleme_egitim = len(y_all)-12
            if izleme_egitim < 30:
                raise ValueError("Tracking Signal için en az 42 aylık geçmiş gerekli.")
            ts_egitim, ts_gercek = y_all[:izleme_egitim], y_all[izleme_egitim:]
            ts_aylar = gr["Ay"].iloc[izleme_egitim:]
            _, _, ts_detay = model_sec_ve_tahmin(ts_egitim, 1)
            ts_aciklama = "Gelecek modunda gerçek verili son 12 ay ayrıca izlenir."
        ts_df = tracking_signal_hesapla(ts_egitim, ts_gercek, ts_aylar, ts_detay)
        ts_son, ts_asim = float(ts_df["TS"].iloc[-1]), int((ts_df["TS"].abs()>4).sum())
        if ts_son>4:
            ts_durum, ts_etiket = "negative", "Eksik tahmin uyarısı"
        elif ts_son<-4:
            ts_durum, ts_etiket = "warning", "Fazla tahmin uyarısı"
        else:
            ts_durum, ts_etiket = "positive", "Sınır içinde"
        chips_goster([
            ("Son TS", f"{ts_son:+.2f}", ts_durum),
            ("Durum", ts_etiket, ts_durum),
            ("±4 dışındaki ay", str(ts_asim), "negative" if ts_asim else "positive"),
            ("İzlenen dönem", str(len(ts_df))+" ay", ""),
        ])
        ciz(ts_grafigi(ts_df), "ts_"+secili+str(egitim_ay)+mod)
        with st.expander("Aylık izleme değerleri ve hesap yöntemi", expanded=False):
            st.caption(ts_aciklama)
            st.dataframe(ts_df.round(3), hide_index=True, use_container_width=True, height=235)
            st.markdown("`TS = RSFE / MAD`; `RSFE = Σ(gerçek − tahmin)`; "
                        "`MAD = Σ|hata| / dönem sayısı`. ±4 uygulamadaki izleme eşiğidir; istatistiksel olarak kalibre edilmiş kontrol limiti değildir.")
    except Exception as e:
        ts_df = pd.DataFrame()
        st.error(f"Tracking Signal hesaplanamadı: {e}")

with sekme_anom:
    st.markdown('<div class="panel-title">Geçmişe dayalı satış anomalileri</div>', unsafe_allow_html=True)
    chips_goster([
        ("Toplam uyarı", str(len(secili_anom)), "negative" if len(secili_anom) else "positive"),
        ("Yüksek satış sıçraması", str(yuksek_sayi), "negative" if yuksek_sayi else "positive"),
        ("Beklenmedik sıfır serisi", str(sifir_sayi), "warning" if sifir_sayi else "positive"),
    ])
    ciz(anomali_grafigi(gr, secili_anom), "anom_"+secili)
    if secili_anom.empty:
        mesaj("Bu üründe tanımlanan eşiklerle anomali uyarısı oluşmadı.", "positive")
    else:
        with st.expander(f"Anomali kayıtları ({len(secili_anom)} kayıt)", expanded=False):
            st.dataframe(secili_anom.round(3), hide_index=True,
                         use_container_width=True, height=245)
    with st.expander("Anomali kuralları ve sınırları", expanded=False):
        st.markdown("**Yüksek satış:** Önceki en fazla 24 ay, en az 12 gözlem ve 6 pozitif satış varsa "
                    "`eşik=max(Q3+1,5×IQR; 1,3×Q3)`. Aşımda kırmızı uyarı.\n\n"
                    "**Sıfır serisi:** Geçmiş satış olasılığı `p=(k+1)/(n+2)`; "
                    "`(1−p)^L < 0,05` ve `L≥2` ise turuncu uyarı. "
                    "Eşikler yalnızca ilgili aydan önceki veriyle hesaplanır.\n\n"
                    "**Değerlendirme:** Uyarılar inceleme gerektirir; anomaliler otomatik silinmez, "
                    "sıfır satış gerçek talep yokluğu ile eşdeğer olmayabilir.")

with sekme_cv:
    st.markdown('<div class="panel-title">Rolling-Origin · aday modellerin doğrulama sıralaması</div>',
                unsafe_allow_html=True)
    chips_goster([
        ("Değerlendirilen model", str(len(cv)), ""),
        ("Seçilen", detay["Model"], "positive"),
        ("CV puanı", tr_num(detay["Puan"], 3), ""),
    ])
    gorunen_sutun = ["Model", "Secim_Puani", "MAE", "RMSE", "WAPE_%", "MASE", "RMSSE", "ME"]
    st.dataframe(cv[gorunen_sutun].round(3), hide_index=True, use_container_width=True, height=320)
    with st.expander("Tüm doğrulama sütunları, sınıflandırma ve atlanan modeller", expanded=False):
        x1, x2 = st.columns([1.6, 1], gap="small")
        with x1:
            st.dataframe(cv.round(3), hide_index=True, use_container_width=True, height=265)
        with x2:
            adi, cv2, sinif = talep_sinifi(y_train)
            chips_goster([("ADI", tr_num(adi, 3), ""),
                          ("CV²", tr_num(cv2, 3), ""), ("Sınıf", sinif, "neutral")])
            st.dataframe(detay["Atlananlar"], hide_index=True,
                         use_container_width=True, height=210)
    if olcum:
        with st.expander("Bağımsız test · model karşılaştırması", expanded=False):
            st.caption("Test verisi model seçiminde kullanılmaz. Bu tabloya bakarak model değiştirmek test tarafsızlığını bozar.")
            test_karsilastirma=[]
            for aday in cv["Model"]:
                try:
                    if aday == "Hibrit_Top2_Esit":
                        t=(temel_tahmin(y_train, detay["EnIyi_Iki"][0],h)+
                           temel_tahmin(y_train, detay["EnIyi_Iki"][1],h))/2
                    elif aday == "Hibrit_Top2_Agirlikli":
                        w=detay["Agirliklar"]
                        t=(w[0]*temel_tahmin(y_train,detay["EnIyi_Iki"][0],h)+
                           w[1]*temel_tahmin(y_train,detay["EnIyi_Iki"][1],h))
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            t=temel_tahmin(y_train,aday,h)
                    m=metrikler(y_test,np.maximum(0,np.asarray(t,float)),y_train)
                    test_karsilastirma.append({"Model":aday,**m})
                except Exception:
                    continue
            if test_karsilastirma:
                st.dataframe(pd.DataFrame(test_karsilastirma).sort_values("MAE").round(3),
                             hide_index=True, use_container_width=True, height=280)
            else:
                st.info("Karşılaştırılabilecek model bulunamadı.")

with sekme_toplu:
    st.markdown('<div class="panel-title">20 / 200 malzeme için toplu Excel raporu</div>',
                unsafe_allow_html=True)
    st.caption("Geçmiş dönem testi, gelecek 12 aylık tahmin, Tracking Signal ve anomali sonuçları tek Excel dosyasında sunulur.")
    min_ay=int(df.groupby("Malzeme Kodu").size().min())
    if min_ay<36:
        st.warning("Bazı malzemelerde 36 aydan az veri var; toplu test için yeterli değil.")
    else:
        ay_max=min_ay-6
        ra, rb = st.columns([1, 1], gap="medium")
        with ra:
            rapor_egitim=st.slider("Eğitim dönemi (ay)", 30, ay_max,
                                   value=min(45,ay_max), key="analiz_toplu_egitim")
        with rb:
            rapor_tam_test=st.checkbox("Kalan tüm ayları test et", value=False, key="analiz_tum_test")
        import hashlib
        imza=(hashlib.sha256(uploaded.getvalue()).hexdigest()
              + f"::{rapor_egitim}::{int(rapor_tam_test)}::v5")
        if st.session_state.get("analiz_v5_rapor_imza")!=imza:
            st.session_state.pop("analiz_v5_rapor_bytes",None)
            st.session_state.pop("analiz_v5_rapor_tablolari",None)
            st.session_state["analiz_v5_rapor_imza"]=imza
        if st.button("Tüm Malzemeleri Analiz Et", type="primary"):
            cubuk=st.progress(0)
            with st.spinner("Ürünlerin tahmin, test, TS ve anomalileri hesaplanıyor..."):
                try:
                    tablolar=toplu_analiz(df,egitim_ay=rapor_egitim,
                        test_tamami=rapor_tam_test,ilerleme=cubuk.progress)
                    st.session_state["analiz_v5_rapor_bytes"]=rapor_excel_uret(tablolar)
                    st.session_state["analiz_v5_rapor_tablolari"]=tablolar
                except Exception as e:
                    st.error(f"Toplu rapor hatası: {type(e).__name__}: {e}")
            cubuk.empty()
        if "analiz_v5_rapor_bytes" in st.session_state:
            tablolar=st.session_state["analiz_v5_rapor_tablolari"]
            ozet=tablolar["Ozet"]
            hatalar=tablolar["Hatalar"]
            chips_goster([
                ("İşlenen",str(len(ozet)),"positive" if len(ozet) else "warning"),
                ("Gelecek tahmini",str(len(tablolar["Gelecek_12_Ay"])),""),
                ("Anomali",str(len(tablolar["Anomali_Detay"])),"warning"),
                ("Hata",str(len(hatalar)),"negative" if len(hatalar) else "positive"),
            ])
            st.download_button("Excel Raporunu İndir", st.session_state["analiz_v5_rapor_bytes"],
                file_name="Talep_Analiz_Raporu.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            if not hatalar.empty:
                mesaj("Bazı malzemeler hesaplanamadı. Hatalar sayfasını kontrol edin.","negative")
            with st.expander("Toplu analiz özeti ve hata kayıtları",expanded=False):
                st.dataframe(ozet.round(3),hide_index=True,use_container_width=True,height=275)
                if not hatalar.empty:
                    st.dataframe(hatalar,hide_index=True,use_container_width=True,height=200)
            rapor_metni=(
                "TALEP ANALİZ VE KARAR DESTEK ÇALIŞMASI\n\n"
                f"İşlenen malzeme: {len(ozet)}\n"
                f"Gelecek 12 ay kayıt sayısı: {len(tablolar['Gelecek_12_Ay'])}\n"
                f"Anomali uyarısı: {len(tablolar['Anomali_Detay'])}\n"
                f"Eğitim: İlk {rapor_egitim} ay\n"
                f"Toplam TS eşik aşımı: {int(ozet['TS_Uyari_Ay'].sum()) if not ozet.empty else 0}\n"
                "\nBu uygulamanın geliştirme ve örnek testleri sentetik verilerle yürütülmüştür. Yüklenen verilerin kaynağı ayrıca doğrulanmalıdır.\n"
            )
            st.download_button("Analiz özetini TXT indir",rapor_metni.encode("utf-8"),
                file_name="Talep_Analiz_Ozeti.txt",mime="text/plain")

with st.expander("Veri ve Yöntem Bilgileri",expanded=False):
    st.caption("Veriler yüklenen Excel dosyasından işlenir. Model seçimine test dönemi dahil edilmez. Anomali uyarıları kayıtları otomatik değiştirmez. Geliştirme testlerinde sentetik satış verisi kullanılmıştır.")
    c1,c2=st.columns([1,1])
    with c1:
        st.dataframe(df,hide_index=True,use_container_width=True,height=225)
    with c2:
        sat=[]
        for kod,g in df.groupby("Malzeme Kodu"):
            adi,cv2,sinif=talep_sinifi(g.sort_values("Ay")["Satis"].to_numpy(float))
            sat.append({"Malzeme Kodu":kod,"ADI":adi,"CV²":cv2,"Talep Sınıfı":sinif})
        st.dataframe(pd.DataFrame(sat).round(3),hide_index=True,
                     use_container_width=True,height=225)

