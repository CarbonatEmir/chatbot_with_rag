import streamlit as st
import streamlit.components.v1 as components
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import base64
import os
import shutil
from pathlib import Path
from sqlalchemy import create_engine, text
from PyPDF2 import PdfReader
import re
import json
from datetime import datetime
import pandas as pd

db_url = "postgresql://postgres:1357913@localhost:5432/lasersan_Ai"
engine = create_engine(db_url)

llm = OllamaLLM(model="qwen2.5", temperature=0)

st.set_page_config(
    page_title="Lasersan AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "show_pdf" not in st.session_state:
    st.session_state.show_pdf = False
if "current_product" not in st.session_state:
    st.session_state.current_product = None
if "aktif_urun" not in st.session_state:
    st.session_state.aktif_urun = ""
if "son_cevap_id" not in st.session_state:
    st.session_state.son_cevap_id = None

page = st.query_params.get("page", "chat")
urun_secimi = st.query_params.get("product")

if urun_secimi != st.session_state.current_product:
    st.session_state.current_product = urun_secimi
    st.session_state.show_pdf = False

def feedback_tablosu_olustur():
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS chatbot_feedback (
                    id SERIAL PRIMARY KEY,
                    soru TEXT,
                    cevap TEXT,
                    rating INT,
                    feedback TEXT,
                    dogru_mu BOOLEAN,
                    tarih TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    cihaz_adi VARCHAR(100),
                    kullanici_duzelti TEXT
                )
            """))
            conn.commit()
    except:
        pass

def feedback_kaydet(soru, cevap, rating, feedback, dogru_mu, cihaz_adi=None, duzelti=None):
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO chatbot_feedback 
                (soru, cevap, rating, feedback, dogru_mu, cihaz_adi, kullanici_duzelti, tarih)
                VALUES (:soru, :cevap, :rating, :feedback, :dogru_mu, :cihaz_adi, :duzelti, :tarih)
            """), {
                "soru": soru,
                "cevap": cevap,
                "rating": rating,
                "feedback": feedback,
                "dogru_mu": dogru_mu,
                "cihaz_adi": cihaz_adi,
                "duzelti": duzelti,
                "tarih": datetime.now()
            })
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Feedback kaydetme hatası: {e}")
        return False

def ogrenme_verilerini_cek():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT soru, cevap, dogru_mu, feedback, kullanici_duzelti 
                FROM chatbot_feedback 
                WHERE dogru_mu = FALSE OR feedback IS NOT NULL
                ORDER BY tarih DESC LIMIT 10
            """)).fetchall()
            return result
    except:
        return []

def pdften_bilgi_cek(pdf_dosyasi):
    try:
        metin = ""
        for sayfa in pdf_dosyasi.pages:
            metin += sayfa.extract_text()
        
        metin_temiz = metin.replace("\n", " ")
        
        prompt = f"""
Aşağıdaki ürün broşürünü analiz et ve cihazın TÜM teknik özelliklerini eksiksiz şekilde çıkar.
SADECE JSON formatında çıktı ver. Başka hiçbir açıklama yapma.

KURALLAR:
1. Temel çerçeve olarak şu anahtarları kesinlikle bulundur: kategori, agirlik, boyut, calisma_sicakligi, fov, kare_hizi, cozunurluk, ip_seviyesi, sensor, lazer_mesafe_olcer, optik_zoom, insan_tespit_menzili, arac_tespit_menzili, pil_omru, donus_acisi, ek_ozellikler.
2. BİLGİ KAYBETME: Broşürde yukarıdaki anahtarlara uymayan ekstra teknik özellikler varsa BUNLARI KESİNLİKLE SİLME. Onlar için JSON içine yeni anahtarlar oluştur ve veriyi ekle.
3. ÇİFT KAMERA: Metinde "Termal Kamera" ve "Düşük Işık Kamerası" gibi iki ayrı bölüm varsa, özellikleri birleştirerek yaz.

Metin:
{metin_temiz[:5000]}
"""
        res = llm.invoke(prompt)
        
        match = re.search(r'\{.*\}', res, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            data = {}
            
        cihaz_adi = st.session_state.aktif_urun or "BILINMIYOR"
        sonuc = {"cihaz_adi": cihaz_adi}
        
        std_keys = ["kategori", "agirlik", "boyut", "calisma_sicakligi", "fov", "kare_hizi", "cozunurluk", "ip_seviyesi", "sensor", "lazer_mesafe_olcer", "optik_zoom", "insan_tespit_menzili", "arac_tespit_menzili", "pil_omru", "donus_acisi", "ek_ozellikler"]
        for k in std_keys:
            sonuc[k] = ""
            
        for k, v in data.items():
            safe_k = re.sub(r'[^a-z0-9_]', '', k.lower().replace(' ', '_'))
            if safe_k and safe_k != "cihaz_adi":
                sonuc[safe_k] = str(v).strip()
            
        return sonuc
    except Exception as e:
        st.error(f"Yapay Zeka Analiz Hatası: {e}")
        return None

def veritabanina_kaydet(bilgiler):
    try:
        with engine.connect() as conn:
            for key in bilgiler.keys():
                safe_key = re.sub(r'[^a-zA-Z0-9_]', '', key.lower())
                conn.execute(text(f"ALTER TABLE cihaz_ozellikleri ADD COLUMN IF NOT EXISTS {safe_key} TEXT"))
            
            existing = conn.execute(
                text("SELECT id FROM cihaz_ozellikleri WHERE cihaz_adi = :adi"),
                {"adi": bilgiler["cihaz_adi"]}
            ).fetchone()
            
            keys = list(bilgiler.keys())
            keys.remove("cihaz_adi")
            
            if existing:
                set_clause = ", ".join([f"{k}=:{k}" for k in keys])
                update_sql = f"UPDATE cihaz_ozellikleri SET {set_clause} WHERE cihaz_adi=:cihaz_adi"
                conn.execute(text(update_sql), bilgiler)
            else:
                all_keys = list(bilgiler.keys())
                cols = ", ".join(all_keys)
                vals = ", ".join([f":{k}" for k in all_keys])
                insert_sql = f"INSERT INTO cihaz_ozellikleri ({cols}) VALUES ({vals})"
                conn.execute(text(insert_sql), bilgiler)
                
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Veritabanı kaydetme hatası: {e}")
        return False

custom_css = """
<style>
    .stApp { background-color: #0f172a !important; }
    @keyframes fadeInSlideUp { 0% { opacity: 0; transform: translateY(15px); } 100% { opacity: 1; transform: translateY(0); } }
    [data-testid="column"] { animation: fadeInSlideUp 0.4s cubic-bezier(0.2, 0.8, 0.2, 1) forwards; }
    [data-testid="stImage"] img { max-height: 380px !important; object-fit: contain !important; margin: 0 auto !important; filter: drop-shadow(0 15px 25px rgba(0,0,0,0.4)); }
    [data-testid="stSidebar"] [data-testid="stImage"] img { max-height: none !important; filter: none !important; }
    .stChatInputContainer > div { border-radius: 24px !important; border: 1px solid #d1d5db !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important; padding-left: 10px; background-color: white; }
    [data-testid="chatAvatarIcon-assistant"] { background-color: #003B70 !important; color: white !important; }
    .prod-title-large { text-align: center; color: #C8102E; font-size: 3.5rem; font-weight: 900; margin-bottom: 20px; font-family: "Segoe UI", sans-serif; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); margin-top: 0; }
    .prod-title-small { text-align: center; color: #C8102E; font-size: 2.2rem; font-weight: 900; margin-bottom: 20px; font-family: "Segoe UI", sans-serif; text-shadow: 1px 1px 3px rgba(0,0,0,0.3); margin-top: 0; }
    .fancy-pdf-header { text-align: center; font-size: 1.6rem; color: #e2e8f0; margin-top: 0; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px dashed #334155; letter-spacing: 2px; font-family: 'Segoe UI', sans-serif; }
    .fancy-pdf-header span { color: #C8102E; font-weight: 900; }
    [data-testid="stSidebar"] { background-color: #44A2BD!important; }
    [data-testid="stSidebar"] [data-testid="stExpander"] { background-color: #1e293b !important; border-radius: 8px !important; border: 1px solid #BFEFF8 !important; margin-bottom: 8px !important; box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important; }
    [data-testid="stSidebar"] [data-testid="stExpander"] details summary { transition: all 0.3s ease !important; padding: 8px !important; border-radius: 8px !important; }
    [data-testid="stSidebar"] [data-testid="stExpander"] details summary:hover { background-color: #334155 !important; }
    [data-testid="stSidebar"] [data-testid="stExpander"] details summary p { color: #e2e8f0 !important; font-weight: 700 !important; font-size: 0.95rem !important; }
    .sidebar-link-btn { display: block; padding: 4px 8px; background-color: #107F93; color: #f8fafc !important; text-align: center; border-radius: 6px; text-decoration: none; font-weight: 700; margin-bottom: 6px; transition: all 0.3s ease; font-size: 0.85rem; border: 1px solid transparent; }
    .sidebar-link-btn:hover { background-color: #003B70; color: white !important; border: 1px solid #003B70; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    button[kind="secondary"] { background-color: #107F93 !important; color: white !important; border-radius: 8px !important; font-weight: 800 !important; font-size: 1.1rem !important; border: none !important; padding: 10px 24px !important; box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important; }
    button[kind="secondary"]:hover { background-color: #003B70 !important; transform: translateY(-2px); }
    button[kind="primary"] { background-color: #C8102E !important; color: white !important; border-radius: 8px !important; font-weight: 800 !important; font-size: 1.1rem !important; border: none !important; padding: 10px 24px !important; box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important; }
    button[kind="primary"]:hover { background-color: #9e0c24 !important; transform: translateY(-2px); }
    .admin-title { text-align: center; color: #C8102E; font-size: 2.8rem; font-weight: 900; margin-bottom: 10px; font-family: "Segoe UI", sans-serif; }
    .admin-subtitle { text-align: center; color: #e2e8f0; font-size: 1.1rem; margin-bottom: 30px; font-family: "Segoe UI", sans-serif; }
    .feedback-box { background: linear-gradient(135deg, #107F93, #44A2BD); padding: 15px; border-radius: 8px; margin-top: 20px; color: white; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

feedback_tablosu_olustur()

with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    logo_file = "logo3.png" if os.path.exists("logo3.png") else "logo.png"
    if os.path.exists(logo_file):
        with open(logo_file, "rb") as f:
            b64_logo = base64.b64encode(f.read()).decode()
        st.markdown(f'''<div style="text-align: center;"><a href="/" target="_self"><img src="data:image/png;base64,{b64_logo}" style="width: 80%; cursor: pointer;"></a></div>''', unsafe_allow_html=True)
           
    st.markdown("<div style='text-align: center; color: #e2e8f0; font-family: \"Segoe UI\", sans-serif; font-size: 0.9rem; font-weight: 700; letter-spacing: 0.5px; margin-top: 10px; margin-bottom: 30px;'>Lasersan Advanced Technology</div>", unsafe_allow_html=True)
    
    st.markdown('<a href="?page=admin" target="_self" style="display: block; padding: 12px; background-color: #C8102E; color: white; text-align: center; border-radius: 8px; text-decoration: none; font-weight: 800; font-size: 1.1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 10px;"> Yeni Ürün Ekle</a>', unsafe_allow_html=True)
    
    st.markdown('<a href="?page=analytics" target="_self" style="display: block; padding: 12px; background-color: #003B70; color: white; text-align: center; border-radius: 8px; text-decoration: none; font-weight: 800; font-size: 1.1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px;"> 📊 Analytics</a>', unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: #f8fafc; font-weight: 800; font-size: 1.2rem; margin-bottom: 15px;'>Ürünler</h3>", unsafe_allow_html=True)
    kategoriler_ve_urunler = {
        "Gece Görüş Sistemleri": ["AURA", "FOCUS", "ODAK", "TUNAY"],
        "Renkli Gece Görüş Sistemleri": ["NOXIS"],
        "Termal El Dürbünü": ["TALOS"],
        "Silah Üstü Nişangahlar": ["KOZGU"],
        "Sürüş Görüş Sistemleri": ["ŞAHAN", "AVCI", "NEBULA", "TOYGAR", "MANTIS", "MUCİZE", "BARBAROS"],
        "Gimbal": ["ALAGÖZ"],
        "Keşif ve Gözetleme Sistemleri": ["YALMAN-150PT", "YALMAN-660PT", "YALMAN-1100PT", "DGV"],
        "Elektro-Optik Nişangahlar": ["DELTA300", "DELTA-180PT", "DELTA-225PT", "DELTA-1100PT"],
        "Radar Sistemleri": ["BARKIN-2D", "KURSAD-20A"],
        "Optik Haberleşme Sistemi": ["RAYPATH"]
    }

    for kategori, urunler in kategoriler_ve_urunler.items():
        with st.expander(kategori):
            for urun_adi in urunler:
                st.markdown(f'<a href="?product={urun_adi}" target="_self" class="sidebar-link-btn">{urun_adi}</a>', unsafe_allow_html=True)
               
    st.markdown("---")
    st.caption("© 2026 Lasersan Savunma Sanayii")

if page == "analytics":
    st.markdown("<h1 class='admin-title'>📊 Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='admin-subtitle'>Chatbot'un öğrenme verilerini görüntüle ve analiz et</p>", unsafe_allow_html=True)
    
    try:
        with engine.connect() as conn:
            total_feedback = conn.execute(text("SELECT COUNT(*) as cnt FROM chatbot_feedback")).fetchone()[0]
            wrong_answers = conn.execute(text("SELECT COUNT(*) as cnt FROM chatbot_feedback WHERE dogru_mu = FALSE")).fetchone()[0]
            accuracy = ((total_feedback - wrong_answers) / total_feedback * 100) if total_feedback > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Toplam Feedback", total_feedback)
            with col2:
                st.metric("Yanlış Cevaplar", wrong_answers)
            with col3:
                st.metric("Doğruluk Oranı", f"{accuracy:.1f}%")
            with col4:
                st.metric("Doğru Cevaplar", total_feedback - wrong_answers)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📉 En Sık Sorulan Cihazlar")
            
            cihaz_stats = conn.execute(text("""
                SELECT cihaz_adi, COUNT(*) as sorulma_sayisi, 
                SUM(CASE WHEN dogru_mu = TRUE THEN 1 ELSE 0 END) as dogru,
                SUM(CASE WHEN dogru_mu = FALSE THEN 1 ELSE 0 END) as yanlis
                FROM chatbot_feedback 
                WHERE cihaz_adi IS NOT NULL
                GROUP BY cihaz_adi 
                ORDER BY sorulma_sayisi DESC LIMIT 10
            """)).fetchall()
            
            if cihaz_stats:
                cihaz_df = pd.DataFrame(cihaz_stats, columns=['Cihaz Adı', 'Sorulma Sayısı', 'Doğru', 'Yanlış'])
                st.dataframe(cihaz_df, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### ❌ Yanlış Cevaplanan Sorular")
            
            yanlis = conn.execute(text("""
                SELECT soru, cevap, feedback, kullanici_duzelti, tarih
                FROM chatbot_feedback 
                WHERE dogru_mu = FALSE 
                ORDER BY tarih DESC LIMIT 10
            """)).fetchall()
            
            if yanlis:
                for i, (soru, cevap, feedback, duzelti, tarih) in enumerate(yanlis, 1):
                    with st.expander(f"{i}. {soru[:50]}... ({tarih.strftime('%Y-%m-%d %H:%M')})"):
                        st.write(f"**Sordu:** {soru}")
                        st.write(f"**Bot Cevabı:** {cevap}")
                        st.write(f"**Kullanıcı Notu:** {feedback or 'Açıklama yok'}")
                        if duzelti:
                            st.write(f"**Doğru Cevap:** {duzelti}")
            else:
                st.info("Henüz yanlış cevaplanan soru yok. Harika! 🎉")
                
    except Exception as e:
        st.error(f"Analytics yükleme hatası: {e}")

elif page == "admin":
    st.markdown("<h1 class='admin-title'> Yönetim Paneli</h1>", unsafe_allow_html=True)
    st.markdown("<p class='admin-subtitle'>PDF broşürünü yükleyin ve veritabanına kaydedin</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        urun_ismi = st.text_input("1. Kaydedilecek Cihazın Adı:", placeholder="Örn: YALMAN-150PT")
        uploaded_file = st.file_uploader("2. PDF Broşürünü Yükleyin:", type="pdf")
        
        if st.button(" Bilgileri Yapay Zeka ile Çek", type="primary", use_container_width=True):
            if not urun_ismi or not uploaded_file:
                st.warning("Lütfen cihaz adını girin ve PDF dosyasını yükleyin.")
            else:
                with st.spinner("Yapay Zeka broşürü analiz ediyor..."):
                    st.session_state.aktif_urun = urun_ismi.strip().upper()
                    pdf_reader = PdfReader(uploaded_file)
                    bilgiler = pdften_bilgi_cek(pdf_reader)
                    if bilgiler:
                        st.session_state.extracted_data = bilgiler

    if "extracted_data" in st.session_state and st.session_state.extracted_data:
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: white; text-align: center; margin-bottom: 20px;'>📋 Çıkarılan Veriler  (Düzenleyebilirsiniz)</h3>", unsafe_allow_html=True)
        
        with st.form("kayit_formu"):
            cols = st.columns(3)
            guncel_bilgiler = {}
            keys = list(st.session_state.extracted_data.keys())
            
            for i, k in enumerate(keys):
                if k == "ek_ozellikler":
                    continue
                c = cols[i % 3]
                with c:
                    label = k.replace("_", " ").title()
                    val = st.session_state.extracted_data[k]
                    guncel_bilgiler[k] = st.text_input(label, value=val if val else "")
            
            st.markdown("<br>", unsafe_allow_html=True)
            guncel_bilgiler["ek_ozellikler"] = st.text_area("Ek Özellikler", value=st.session_state.extracted_data.get("ek_ozellikler") or "", height=100)
            
            st.markdown("#### ➕ Yeni Özellik Ekle ")
            col_yeni1, col_yeni2 = st.columns(2)
            with col_yeni1:
                yeni_oz_ad = st.text_input("Sütun Adı (Örn: Lazer İşaretleyici)")
            with col_yeni2:
                yeni_oz_deger = st.text_input("Değeri (Örn: Var)")

            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("✅ Onayla ve Veritabanına Kaydet", use_container_width=True)
            
            if submit:
                if yeni_oz_ad and yeni_oz_deger:
                    turkce_karakterler = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
                    safe_col_name = yeni_oz_ad.translate(turkce_karakterler).lower().replace(" ", "_")
                    safe_col_name = re.sub(r'[^a-z0-9_]', '', safe_col_name)
                    
                    if safe_col_name:
                        guncel_bilgiler[safe_col_name] = yeni_oz_deger
                
                if veritabanina_kaydet(guncel_bilgiler):
                    st.balloons()
                    st.success(f"Tebrikler! {guncel_bilgiler['cihaz_adi']} veritabanına işlendi.")
                    del st.session_state.extracted_data

elif urun_secimi:
    resim_yolu = f"{urun_secimi}.png"
    pdf_yolu = f"urunpdf/{urun_secimi}.pdf"

    if not st.session_state.show_pdf:
        st.markdown("<br><br>", unsafe_allow_html=True)
        col_bosluk1, col_main, col_bosluk2 = st.columns([1, 1, 1])
        with col_main:
            st.markdown(f"<div class='prod-title-large'>{urun_secimi}</div>", unsafe_allow_html=True)
            if os.path.exists(resim_yolu): st.image(resim_yolu, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f" Broşürü Görüntüle", use_container_width=True):
                st.session_state.show_pdf = True
                st.rerun()
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        col_sol, col_sag = st.columns([1, 2.2])
        with col_sol:
            st.markdown("""<style>[data-testid="column"]:nth-child(1) [data-testid="stImage"] img { max-height: 250px !important; }</style>""", unsafe_allow_html=True)
            st.markdown(f"<div class='prod-title-small'>{urun_secimi}</div>", unsafe_allow_html=True)
            if os.path.exists(resim_yolu): st.image(resim_yolu, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Broşürü Kapat", type="primary", use_container_width=True):
                st.session_state.show_pdf = False
                st.rerun()

            if os.path.exists(pdf_yolu):
                with open(pdf_yolu, "rb") as f: pdf_base64 = base64.b64encode(f.read()).decode('utf-8')
                js_code = f"""
                <!DOCTYPE html>
                <html><head><style>body {{ margin: 0; padding: 0; background: transparent; display: flex; justify-content: center; }} .tab-btn {{ background-color: #334155; color: #ffffff; padding: 10px; border-radius: 8px; font-weight: 350; font-size: 1.1rem; border: none; cursor: pointer; width: 100%; font-family: 'Segoe UI', sans-serif; transition: all 0.3s; box-shadow: 0 4px 6px rgba(0,0,0,0.2); margin-top: 5px; }} .tab-btn:hover {{ background-color: #1e293b; transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.4); }}</style></head>
                <body><button onclick="openPdf()" class="tab-btn"> Yeni Sekmede Aç</button>
                <script>function openPdf() {{ var b64Data = '{pdf_base64}'; var byteCharacters = atob(b64Data); var byteNumbers = new Array(byteCharacters.length); for (var i = 0; i < byteCharacters.length; i++) {{ byteNumbers[i] = byteCharacters.charCodeAt(i); }} var byteArray = new Uint8Array(byteNumbers); var file = new Blob([byteArray], {{ type: 'application/pdf' }}); var fileURL = URL.createObjectURL(file); window.open(fileURL, '_blank'); }}</script></body></html>
                """
                components.html(js_code, height=60)
        with col_sag:
            if os.path.exists(pdf_yolu):
                st.markdown(f"<div class='fancy-pdf-header'><span>{urun_secimi}</span> ÜRÜN BROŞÜRÜ</div>", unsafe_allow_html=True)
                pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}#toolbar=0&navpanes=0&scrollbar=0" width="100%" height="800" type="application/pdf" style="border-radius: 12px; border: 2px solid #107F93; box-shadow: 0 8px 25px rgba(0,0,0,0.6);"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)

else:
    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        logo_yolu = "logo3.png" if os.path.exists("logo3.png") else "logo.png"
        if os.path.exists(logo_yolu): st.image(logo_yolu, use_container_width=True)

    st.markdown("<h1 class='main-title'><span style='color: #003B70;'>Laser</span><span style='color: #C8102E;'>san Akıllı Asistan</span></h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Tüm şirket cihazları hakkında anında ve güvenilir bilgi alın.</p>", unsafe_allow_html=True)

    @st.cache_resource
    def sistemi_hazirla():
        if Path("chroma_db").exists():
            shutil.rmtree("chroma_db")
       
        loader = PyPDFDirectoryLoader("urunpdf/")
        ham_belgeler = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        belgeler = text_splitter.split_documents(ham_belgeler)
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        vectorstore = Chroma.from_documents(documents=belgeler, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        return retriever
       
    retriever = sistemi_hazirla()
    
    prompt = ChatPromptTemplate.from_template(
        "Sen Lasersan Savunma Sanayii'nin yapay zeka asistanısın. Müşterilere net, kısa ve insan gibi cevap vermelisin.\n\n"
        "<KURALLAR>\n"
        "1. YALNIZCA TÜRKÇE yanıt ver.\n"
        "2. YASAKLI İFADELER: Cümleye 'Müşteri için', 'Size yardımcı olayım' gibi kalıplarla BAŞLAMAK KESİNLİKLE YASAKTIR. Doğrudan insan gibi konuya gir.\n"
        "3. MATEMATİKSEL FİLTRELEME: Müşteri bir sayı belirterek şart koşarsa, SADECE şartı sağlayanları listele. Diğerlerini ASLA gösterme.\n"
        "4. {dinamik_uyari}\n"
        "</KURALLAR>\n\n"
        "<VERITABANI>\n{db_context}\n</VERITABANI>\n\n"
        "<BROSUR>\n{context}\n</BROSUR>\n\n"
        "<GECMIS>\n{chat_history}\n</GECMIS>\n\n"
        "Müşteri: {input}\n"
        "Asistan:"
    )
   
    def kosullu_baglam_getir(inputs):
        if inputs.get("bypass_pdf", False):
            return "LİSTELEME/FİLTRELEME SORGUSU. SADECE VERİTABANINA BAKINIZ."
        docs = retriever.invoke(inputs["arama_metni"])
        return "\n\n".join([doc.page_content for doc in docs])
   
    rag_zinciri = (
        {
            "context": kosullu_baglam_getir,
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history"),
            "db_context": itemgetter("db_context"),
            "dinamik_uyari": itemgetter("dinamik_uyari"),
            "bypass_pdf": itemgetter("bypass_pdf")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    if "mesajlar" not in st.session_state:
        st.session_state.mesajlar = []

    if len(st.session_state.mesajlar) == 0:
        st.markdown("<p class='welcome-text'>Bugün size nasıl yardımcı olabilirim?</p>", unsafe_allow_html=True)

    for mesaj in st.session_state.mesajlar:
        with st.chat_message(mesaj["rol"]):
            st.markdown(mesaj["icerik"])

    soru = st.chat_input("Lasersan cihazlarıyla ilgili bir soru sorun...")

    if soru:
        with st.chat_message("user"):
            st.markdown(soru)
       
        with st.chat_message("assistant"):
            cevap_alani = st.empty()
            cevap_alani.markdown("Düşünüyor ve veritabanını tarıyor...")
           
            gecmis_metni = ""
            for m in st.session_state.mesajlar[-2:]: 
                kim = "Müşteri" if m["rol"] == "user" else "Asistan"
                gecmis_metni += f"{kim}: {m['icerik']}\n"
               
            urun_listesi = ["AURA", "FOCUS", "ODAK", "TUNAY", "NOXIS", "TALOS", "KOZGU", "ŞAHAN", "AVCI", "NEBULA", "TOYGAR", "MANTIS", "MUCİZE", "ALAGÖZ", "YALMAN-150PT", "YALMAN-660PT", "YALMAN-1100PT", "DGV", "DELTA300", "DELTA-180PT", "DELTA-225PT", "BARKIN-2D", "KURSAD-20A", "RAYPATH", "BARBAROS"]
           
            soru_kucuk = soru.lower()
            
            liste_ve_kiyas_kelimeleri = ['listele', 'say', 'kaç', 'isimleri', 'neler', 'hangileri', 'hangi', 'büyük', 'küçük', 'fazla', 'az', 'kilo', 'ağır', 'hafif', 'yüksek', 'düşük', 'altında', 'üstünde', 'kadar', 'tümü', 'tümünü', '<', '>', '=']
            liste_sorgusu = any(kelime in soru_kucuk for kelime in liste_ve_kiyas_kelimeleri)
           
            bahsedilen_urun = None
            for urun in urun_listesi:
                if urun.lower() in soru_kucuk:
                    bahsedilen_urun = urun
                    break
                    
            if bahsedilen_urun:
                st.session_state.aktif_urun = bahsedilen_urun
            elif not liste_sorgusu:
                for m in reversed(st.session_state.mesajlar):
                    if m["rol"] == "user":
                        for urun in urun_listesi:
                            if urun.lower() in m["icerik"].lower():
                                bahsedilen_urun = urun
                                st.session_state.aktif_urun = bahsedilen_urun
                                break
                    if bahsedilen_urun:
                        break
            
            arama_metni = soru
            dinamik_uyari = ""
            
            if liste_sorgusu:
                arama_metni = "BOŞ ARAMA" 
                dinamik_uyari = "Kullanıcı listeleme istiyor. SADECE şartı sağlayanları yaz."
                hedef_cihaz = "TÜMÜ"
            elif st.session_state.aktif_urun:
                arama_metni = f"{st.session_state.aktif_urun} {soru}"
                dinamik_uyari = f"Müşteri {st.session_state.aktif_urun} cihazını soruyor."
                hedef_cihaz = st.session_state.aktif_urun
            else:
                dinamik_uyari = "Genel bir soru soruluyor."
                hedef_cihaz = "TÜMÜ"

            db_context_str = "Sistem Veritabanına Bağlanılamadı."
            try:
                with engine.connect() as conn:
                    if hedef_cihaz == "TÜMÜ":
                        query = text("SELECT * FROM cihaz_ozellikleri")
                        results = conn.execute(query).mappings().fetchall()
                        baslik = "=== VERİTABANINDAKİ CİHAZLARIN LİSTESİ ===\n\n"
                    else:
                        query = text("SELECT * FROM cihaz_ozellikleri WHERE cihaz_adi = :urun")
                        results = conn.execute(query, {"urun": hedef_cihaz.upper()}).mappings().fetchall()
                        baslik = f"=== {hedef_cihaz.upper()} CİHAZI ===\n\n"
                    
                    if results:
                        db_context_str = baslik
                        for r in results:
                            r_dict = dict(r)
                            cihaz_adi = r_dict.pop('cihaz_adi', 'DEGER_YOK')
                            db_context_str += f"CİHAZ ADI: {cihaz_adi}\n"
                            r_dict.pop('id', None)
                            
                            for k, v in r_dict.items():
                                str_val = str(v) if v and str(v).strip() not in ["Belirtilmemiş", "Yok"] else "DEGER_YOK"
                                readable_key = k.replace("_", " ").title()
                                db_context_str += f"- {readable_key}: {str_val}\n"
                            
                            db_context_str += "--------------------------------------\n"
                    else:
                        db_context_str = "Veritabanında cihaz kaydı bulunmamaktadır."
            except Exception as e:
                st.error(f"Veritabanı Hatası: {e}")

            cevap = rag_zinciri.invoke({
                "input": soru,
                "arama_metni": arama_metni,
                "chat_history": gecmis_metni,
                "db_context": db_context_str,
                "dinamik_uyari": dinamik_uyari,
                "bypass_pdf": liste_sorgusu
            })
            
            st.session_state.son_cevap_id = len(st.session_state.mesajlar)
            cevap_alani.markdown(cevap)
           
            st.markdown("<div class='feedback-box'><h4>Bu cevap yardımcı oldu mu?</h4></div>", unsafe_allow_html=True)
            
            col_fb1, col_fb2, col_fb3, col_fb4 = st.columns(4)
            with col_fb1:
                if st.button("👍 Evet", key=f"yes_{len(st.session_state.mesajlar)}"):
                    feedback_kaydet(soru, cevap, 5, "", True, st.session_state.aktif_urun if st.session_state.aktif_urun else None)
                    st.success("Teşekkürler! Feedback kaydedildi.")
                    st.rerun()
            
            with col_fb2:
                if st.button("👎 Hayır", key=f"no_{len(st.session_state.mesajlar)}"):
                    st.session_state.show_feedback_form = True
            
            with col_fb3:
                if st.button("⭐ 5 Yıldız", key=f"star_{len(st.session_state.mesajlar)}"):
                    feedback_kaydet(soru, cevap, 5, "Mükemmel cevap!", True, st.session_state.aktif_urun if st.session_state.aktif_urun else None)
                    st.success("Mükemmel! Teşekkürler.")
                    st.rerun()
            
            if st.session_state.get("show_feedback_form", False):
                with st.form("feedback_form"):
                    feedback_text = st.text_area("Lütfen hangi kısımda yanlış olduğunu açıkla:")
                    dogrulmasi = st.text_area("Doğru cevap ne olmalıydı?:")
                    
                    if st.form_submit_button("Gönder"):
                        feedback_kaydet(soru, cevap, 2, feedback_text, False, st.session_state.aktif_urun if st.session_state.aktif_urun else None, dogrulmasi)
                        st.success("Feedback gönderildi! Chatbot'unuz bu bilgilerden öğrenecek. 🚀")
                        st.session_state.show_feedback_form = False
                        st.rerun()
           
        st.session_state.mesajlar.append({"rol": "user", "icerik": soru})
        st.session_state.mesajlar.append({"rol": "assistant", "icerik": cevap})