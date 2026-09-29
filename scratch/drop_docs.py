import psycopg

DATABASE_URL = "postgresql://postgres:postgres@localhost:5433/postgres"

def drop_docs():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS documents CASCADE")
            cur.execute("DROP TABLE IF EXISTS templates CASCADE")
            cur.execute("DROP TABLE IF EXISTS filled_forms CASCADE")
            conn.commit()
    print("Tables dropped successfully.")

if __name__ == "__main__":
    drop_docs()
