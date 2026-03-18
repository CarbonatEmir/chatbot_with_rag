import sqlalchemy
from sqlalchemy import create_engine, text

db_url = "postgresql://postgres:1357913@localhost:5432/lasersan_Ai"

try:
    print("Veritabanına bağlanmaya çalışılıyor...")
    engine = create_engine(db_url)
    
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))
        print("\n BAĞLANTI BAŞARILI!")
        print("PostgreSQL Versiyonu:", result.fetchone()[0])
        
except Exception as e:
    print("\n Bağlantı hatası oluştu:")
    print(e)