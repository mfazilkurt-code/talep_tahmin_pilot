# Talep Analiz ve Karar Destek Sistemi

Excel üzerinden aylık satış verisi yükleyerek talep sınıflandırması, tahmin, geçmiş dönem testi, Tracking Signal ve anomali analizleri sunan Streamlit uygulaması.

## Yerel kullanım

```bash
pip install -r requirements.txt
streamlit run app_anonim.py
```

## Streamlit Community Cloud

1. Bu klasörün **içindeki dosyaları** GitHub deposunun köküne koyun.
2. `https://share.streamlit.io/` üzerinden GitHub ile oturum açın.
3. `Create app` > `Yup, I have an app` seçin.
4. Repository ve `main` branch'i seçin.
5. Main file path: `app_anonim.py`
6. `Deploy` düğmesine basın.

Kullanıcı uygulamada `.xlsx` dosyasını doğrudan yükler. Satış verilerini GitHub deposuna koymanız gerekmez.

**Not:** Gerçek kurum verilerini yalnızca kullanım ve paylaşım yetkiniz varsa uygulamaya yükleyin. Herkese açık bir uygulamaya gizli veri yüklemeden önce kurumun veri güvenliği ve bulut kullanım kurallarını kontrol edin.
