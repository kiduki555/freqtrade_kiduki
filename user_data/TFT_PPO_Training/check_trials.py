import sqlite3

conn = sqlite3.connect('logs/optuna_studies/ppo_tft.db')
cursor = conn.cursor()

# 테이블 구조 확인
cursor.execute("PRAGMA table_info(trials)")
columns = cursor.fetchall()
print("Trial table columns:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

# 최근 시행들 확인
cursor.execute('SELECT * FROM trials ORDER BY trial_id DESC LIMIT 5')
print('\nRecent trials:')
for row in cursor.fetchall():
    print(f"Trial {row[0]}: {row}")

# 전체 시행 수 확인
cursor.execute('SELECT COUNT(*) FROM trials')
total_trials = cursor.fetchone()[0]
print(f'\nTotal trials: {total_trials}')

# trial_values 테이블 확인
cursor.execute("PRAGMA table_info(trial_values)")
value_columns = cursor.fetchall()
print("\nTrial values table columns:")
for col in value_columns:
    print(f"  {col[1]} ({col[2]})")

# 최고 값들 확인
cursor.execute('SELECT trial_id, value FROM trial_values ORDER BY value DESC LIMIT 10')
print('\nBest trial values:')
for row in cursor.fetchall():
    print(f"Trial {row[0]}: value={row[1]}")

conn.close()
