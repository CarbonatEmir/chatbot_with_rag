from sqlalchemy import create_engine, text

db_url = "postgresql://postgres:1357913@localhost:5432/lasersan_Ai"
engine = create_engine(db_url)

tablo_olustur_sql = """
CREATE TABLE IF NOT EXISTS cihaz_ozellikleri (
    id SERIAL PRIMARY KEY,
    cihaz_adi VARCHAR(100) UNIQUE NOT NULL,
    kategori VARCHAR(100),
    agirlik VARCHAR(50),
    boyut VARCHAR(100),
    calisma_sicakligi VARCHAR(50),
    fov VARCHAR(50),
    kare_hizi VARCHAR(50),
    ek_ozellikler TEXT
);
"""

veri_ekle_sql = """
INSERT INTO cihaz_ozellikleri (cihaz_adi, kategori, agirlik, boyut, calisma_sicakligi, fov, kare_hizi, ek_ozellikler)
VALUES 
('ODAK', 'Gece Görüş Sistemleri', '470 gram', '110x97x90mm', '-32°C ~ +55°C', '43° ±2%', 'Broşürde belirtilmemiş', 'Su geçirmezlik: IP67'),
('AURA', 'Gece Görüş Sistemleri', '500 gram', '120x100x85mm', '-30°C ~ +50°C', '40°', '50 Hz', 'Kompakt tasarım'),
('ALAGÖZ', 'Gimbal', '2.5 kg', '200x150x250mm', '-40°C ~ +60°C', '1.59° ~ 2.5°', '5~60 fps', 'Lazer mesafe ölçer entegreli')
ON CONFLICT (cihaz_adi) DO NOTHING;
"""

try:
    with engine.connect() as conn:
        conn.execute(text(tablo_olustur_sql))

        conn.execute(text(veri_ekle_sql))
        conn.commit()
        print(" Başarılı")
except Exception as e:
    print(" hata ", e)