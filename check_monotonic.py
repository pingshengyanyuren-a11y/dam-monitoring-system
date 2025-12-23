import sqlite3

DB_PATH = 'data/processed/predictions.db'

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

node_id = 200  # 示例节点，可自行更改
rows = cursor.execute('SELECT time_step, final_pred FROM predictions WHERE node_id=? ORDER BY time_step', (node_id,)).fetchall()

prev = None
non_monotonic = False
for t, pred in rows:
    # final_pred 为累计沉降（mm），数值越负表示沉降越大，理论上应单调递减（数值越负）
    if prev is not None and pred > prev:
        non_monotonic = True
        print(f'Non-monotonic at time {t}: prev {prev:.2f} mm, curr {pred:.2f} mm')
        break
    prev = pred

print('Total records checked:', len(rows))
print('Monotonic check passed' if not non_monotonic else 'Monotonic check failed')
conn.close()
