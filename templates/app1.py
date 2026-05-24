import io
import os
import sqlite3
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_file, session
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局变量存储当前 DataFrame
current_df = None
current_filename = None

# ==================== 数据库初始化 ====================
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  filename TEXT,
                  analysis_type TEXT,
                  result TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

init_db()

# ==================== 辅助函数 ====================
def get_data():
    """安全获取当前数据"""
    global current_df
    if current_df is None:
        raise ValueError("无数据，请先上传文件")
    return current_df

def save_history(analysis_type, result=""):
    """记录分析历史"""
    if 'user_id' in session:
        conn = sqlite3.connect('database.db')
        conn.execute(
            "INSERT INTO history (user_id, filename, analysis_type, result) VALUES (?, ?, ?, ?)",
            (session['user_id'], current_filename or 'unknown', analysis_type, result)
        )
        conn.commit()
        conn.close()

# ==================== 页面路由 ====================
@app.route('/')
def index():
    return render_template('index.html')

# ==================== 1. 文件上传 ====================
@app.route('/upload', methods=['POST'])
def upload_file():
    global current_df, current_filename
    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            return jsonify({'error': '未选择文件'}), 400

        current_filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 读取文件
        if file.filename.lower().endswith('.csv'):
            try:
                current_df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                current_df = pd.read_csv(filepath, encoding='gbk')
        else:
            current_df = pd.read_excel(filepath)

        # 记录历史
        save_history('upload')

        preview = current_df.head(10).fillna('').to_dict(orient='records')
        return jsonify({
            'columns': current_df.columns.tolist(),
            'preview': preview,
            'shape': [len(current_df), len(current_df.columns)]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

# ==================== 2. 数据清洗 ====================
@app.route('/clean', methods=['POST'])
def clean_data():
    global current_df
    try:
        data = request.get_json() or {}
        method = data.get('method', 'median')
        z_threshold = float(data.get('z_threshold', 3))

        df = get_data()
        initial_rows = len(df)
        missing_before = df.isnull().sum().sum()

        # 缺失值处理
        if method == 'drop':
            df = df.dropna()
        elif method == 'mean':
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    mode_vals = df[col].mode()
                    if not mode_vals.empty:
                        df[col].fillna(mode_vals[0], inplace=True)
        else:  # median (默认)
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    mode_vals = df[col].mode()
                    if not mode_vals.empty:
                        df[col].fillna(mode_vals[0], inplace=True)

        # Z-score 异常值检测
        numeric_cols = df.select_dtypes(include='number').columns
        outlier_count = 0
        for col in numeric_cols:
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std > 0:
                z_scores = np.abs((df[col] - col_mean) / col_std)
                outlier_count += (z_scores > z_threshold).sum()

        current_df = df
        missing_after = df.isnull().sum().sum()
        filled_count = missing_before - missing_after

        message = f"清洗完成：填充了 {filled_count} 个缺失值，检测到 {outlier_count} 个可能的异常值（Z-score > {z_threshold}）"
        save_history('clean', message)

        preview = current_df.head(10).fillna('').to_dict(orient='records')
        return jsonify({
            'message': message,
            'preview': preview
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== 3. 图表数据 ====================
@app.route('/chart', methods=['GET'])
def chart_data():
    try:
        x_col = request.args.get('x')
        y_col = request.args.get('y')
        df = get_data()

        if x_col not in df.columns or y_col not in df.columns:
            return jsonify({'error': '列名不存在'}), 400

        clean = df[[x_col, y_col]].dropna()
        # 尝试转换为数值，如果失败则跳过
        chart_points = []
        for _, row in clean.iterrows():
            try:
                chart_points.append([float(row[x_col]), float(row[y_col])])
            except (ValueError, TypeError):
                continue

        return jsonify({'data': chart_points})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== 4. 聚类分析（双算法对比） ====================
@app.route('/cluster', methods=['POST'])
def cluster_analysis():
    global current_df
    try:
        df = get_data()
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) < 2:
            return jsonify({'error': '至少需要2个数值列'}), 400

        num_data = df[numeric_cols].dropna()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(num_data)

        # 取前两列作为绘图坐标
        plot_x = scaled[:, 0].tolist()
        plot_y = scaled[:, 1].tolist()

        results = {'scores': {}, 'kmeans': [], 'dbscan': []}

        # K-Means
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(scaled)
        results['scores']['K-Means'] = round(silhouette_score(scaled, kmeans_labels), 4)
        results['kmeans'] = [[float(plot_x[i]), float(plot_y[i])] for i in range(len(plot_x))]

        # DBSCAN
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(scaled)
            unique_labels = set(dbscan_labels)
            if len(unique_labels) > 1 and -1 not in unique_labels:
                # 有多个有效簇
                score = silhouette_score(scaled, dbscan_labels)
            elif len(unique_labels) > 1:
                # 有噪声点，只在非噪声点上计算
                mask = dbscan_labels != -1
                if mask.sum() > 1 and len(set(dbscan_labels[mask])) > 1:
                    score = silhouette_score(scaled[mask], dbscan_labels[mask])
                else:
                    score = 0.0
            else:
                score = 0.0
            results['scores']['DBSCAN'] = round(score, 4)
            results['dbscan'] = [[float(plot_x[i]), float(plot_y[i])] for i in range(len(plot_x))]
        except Exception as e:
            results['scores']['DBSCAN'] = 0.0
            results['dbscan'] = [[float(plot_x[i]), float(plot_y[i])] for i in range(len(plot_x))]

        # 把聚类标签加入数据
        current_df['cluster_kmeans'] = kmeans_labels

        save_history('clustering', f"K-Means:{results['scores']['K-Means']}, DBSCAN:{results['scores']['DBSCAN']}")

        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== 5. 用户注册 ====================
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()

        if not username or not password:
            return jsonify({'status': 'error', 'message': '用户名和密码不能为空'})

        conn = sqlite3.connect('database.db')
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            return jsonify({'status': 'success'})
        except sqlite3.IntegrityError:
            return jsonify({'status': 'error', 'message': '用户名已存在'})
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ==================== 6. 用户登录 ====================
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()

        conn = sqlite3.connect('database.db')
        user = conn.execute(
            "SELECT id, username FROM users WHERE username=? AND password=?",
            (username, password)
        ).fetchone()
        conn.close()

        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': '用户名或密码错误'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ==================== 7. 历史记录 ====================
@app.route('/history', methods=['GET'])
def get_history():
    if 'user_id' not in session:
        return jsonify({'error': '请先登录'})

    conn = sqlite3.connect('database.db')
    records = conn.execute(
        "SELECT filename, analysis_type, result, timestamp FROM history WHERE user_id=? ORDER BY timestamp DESC LIMIT 20",
        (session['user_id'],)
    ).fetchall()
    conn.close()

    return jsonify([
        {'filename': r[0], 'type': r[1], 'time': r[3]}
        for r in records
    ])

# ==================== 8. 文件导出 ====================
@app.route('/export', methods=['GET'])
def export_data():
    global current_df
    if current_df is None:
        return jsonify({'error': '无数据可导出'}), 400

    output = io.StringIO()
    current_df.to_csv(output, index=False, encoding='utf-8-sig')
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8-sig')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='processed_data.csv'
    )

# ==================== 启动 ====================
if __name__ == '__main__':
    app.run(debug=True)